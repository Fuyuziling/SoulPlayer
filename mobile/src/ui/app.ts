import './styles.css';
import { SensorEngine } from '../core/engine';
import { MusicMatcher, type MusicGroup } from '../core/music-match';
import { createMotionSource } from '../platform/motion';
import { Player, type Track, type TrackGroup } from '../platform/player';
import { Scope } from './scope';
import * as store from '../platform/storage';
import { actionLabel, modesOf, MODES } from '../core/presets';
import type { ActionEvent, FrameDiagnostics, ModeId, Placement } from '../core/types';

type Tab = 'gestures' | 'library' | 'settings';

function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  cls?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  if (text !== undefined) el.textContent = text;
  return el;
}

function fmtTime(sec: number): string {
  if (!Number.isFinite(sec) || sec < 0) return '0:00';
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

/** 文件夹名来自契约（唯一真源），UI 不自己写死中文。 */
function folderLabel(g: MusicGroup): string {
  return MODES.modes.run.music?.folders.find((f) => f.id === g)?.label ?? g;
}

export class App {
  private engine: SensorEngine;
  private motion = createMotionSource();
  private player = new Player();
  private scope: Scope | null = null;

  private root: HTMLElement;
  private settings = store.loadSettings();
  private tracks: Track[] = [];
  private tab: Tab = 'gestures';

  private dirty = true;
  /** 启动后收到的样本数。用来判断"通道开了但没数据"，这是真机上最常见的故障。 */
  private sampleCount = 0;
  /** 最近一帧节律是否成立。芯片文字用它区分"跑步"与"跑步中"。 */
  private lastRunning = false;
  /** 日常模式下检测到跑步，芯片持续高亮直到用户做选择。 */
  private suggested = false;
  /** 最近一次渲染时正在播放的曲目 id。变了才刷新曲库页的"播放中"指针。 */
  private lastTrackId: string | null = null;
  /** 硬件步检测是否可用（机型 + 权限都满足）。设置页展示用。 */
  private stepDetectorOn = false;
  /** 跑步模式的步频乐单匹配器。非跑步模式下为 null，不消耗任何计算。 */
  private musicMatcher: MusicMatcher | null = null;
  /** 挂起的乐单切换：当前歌剩余不足 60s 时等它播完再执行。 */
  private pendingSwitch: MusicGroup | null = null;
  /** 淡出切换进行中。防止两段淡出交叠把音量刷乱。 */
  private fading = false;
  /** 空组提醒去重：每组只提示一次，不然跑步中每隔一段就弹一遍很烦。 */
  private emptyWarned = new Set<MusicGroup>();
  /** 曲库页当前打开的文件夹。null = 三个文件夹总览，'all' = 全部曲目。 */
  private libraryFolder: MusicGroup | 'all' | null = null;
  private raf = 0;
  private toastTimer = 0;

  // 缓存 DOM 引用，避免每帧 querySelector
  private els: Record<string, HTMLElement> = {};

  constructor(root: HTMLElement) {
    this.root = root;
    this.engine = new SensorEngine({
      noiseFloor: this.settings.noiseFloor,
      mode: this.settings.mode,
      placement: this.settings.placement,
    });
    this.engine.setSensitivity(this.settings.sensitivity);
    for (const [id, on] of Object.entries(this.settings.enabledPresets)) {
      this.engine.setPresetEnabled(id, on);
    }

    this.motion.onSample = (s) => {
      // ingest() 返回补齐了去重力结果和重力估计的样本，示波器直接用它，
      // 不必再维护一份缓冲——两处缓冲一旦不同步，画出来的波形和判定用的数据就不是一回事了。
      const sample = this.engine.ingest(s);
      this.sampleCount++;
      this.scope?.push(sample);
      this.dirty = true;
    };
    // 硬件步检测（TYPE_STEP_DETECTOR）：跑步提示的第二确认源。
    // 不可用时引擎自动中性降级，这里不需要任何分支。
    this.motion.onStep = (t) => this.engine.noteStep(t);
    // 授权结果必须让用户看得见：弹窗确认后界面毫无反馈，
    // 用户无法分辨"授权生效了"还是"功能根本没做"（真机反馈踩过的坑）。
    this.motion.onStepDetector = (available) => {
      const prev = this.stepDetectorOn;
      this.engine.setStepDetectorAvailable(available);
      this.stepDetectorOn = available;
      if (available && !prev) this.showToast('步检测已启用 · 跑步识别更准');
      if (this.tab === 'settings') this.renderSheet();
    };
  }

  async boot(): Promise<void> {
    this.tracks = await store.listTracks().catch(() => []);
    await this.attachTrackUrls();
    this.player.setTracks(this.tracks);
    this.player.setVolume(this.settings.volume);
    this.player.onState = () => this.renderNow();
    this.player.onEnded = () => {
      // 折中切换规则的后半段：剩余不足 60s 时挂起的切换，在这里兑现。
      // 必须排在 next() 之前——先把队列换成目标组，随机起一首，
      // 否则 next() 还是旧队列的下一首。
      if (this.pendingSwitch) {
        const g = this.pendingSwitch;
        this.pendingSwitch = null;
        this.applyRunGroup(g, true);
        return;
      }
      void this.player.next();
    };
    this.renderGate();
  }

  // ── 启动门 ──────────────────────────────

  private renderGate(): void {
    this.root.innerHTML = '';
    const gate = h('div', 'gate');
    gate.style.animationDelay = '0s';

    const title = h('h1', '', 'SoulPlayer');
    const lead = h(
      'p',
      'lead',
      '不用看屏幕，也不用碰按钮。甩一下换歌，把手机翻过来暂停，摇一摇随机。全部靠手机里的加速度计和陀螺仪。',
    );

    const rules = h('div', 'rules');
    const ruleData: Array<[string, string, string]> = [
      ['↔', '向左 / 向右甩', '上一首 / 下一首'],
      ['⟳', '翻面（屏幕朝下）', '暂停或继续'],
      ['≈', '摇一摇', '随机一首'],
      ['◑', '向右 / 向左倾斜保持', '音量增大 / 减小'],
    ];
    for (const [ico, t, d] of ruleData) {
      const row = h('div', 'rule');
      row.appendChild(h('div', 'ico', ico));
      const box = h('div');
      box.appendChild(h('div', 't', t));
      box.appendChild(h('div', 'd', d));
      row.appendChild(box);
      rules.appendChild(row);
    }

    const btn = h('button', 'btn accent', '启用传感器并开始');
    btn.style.marginTop = '6px';
    btn.onclick = () => void this.start(btn, statusEl);

    const statusEl = h('div', 'hint', '首次使用需要授权动作传感器。数据不会离开这台手机。');
    const secure = h(
      'div',
      'hint',
      window.isSecureContext
        ? '提示：把这个页面加到主屏幕，可以全屏使用。'
        : '⚠ 当前不是 https 环境，浏览器会拒绝传感器访问，请使用线上链接打开。',
    );
    if (!window.isSecureContext) secure.classList.add('warn');

    gate.append(title, lead, rules, btn, statusEl, secure);
    this.root.appendChild(gate);
  }

  private async start(btn: HTMLButtonElement, statusEl: HTMLElement): Promise<void> {
    btn.disabled = true;
    statusEl.textContent = '正在请求权限…';

    // 必须在用户手势里解锁音频，否则后续手势触发的播放会被浏览器静默拒绝
    this.player.unlock();

    const st = await this.motion.start();
    if (!st.active) {
      btn.disabled = false;
      statusEl.textContent = st.message || '传感器启用失败';
      statusEl.classList.add('warn');
      return;
    }

    statusEl.textContent = '请把手机平稳放在手上，保持 1.5 秒不要动，正在校准…';
    this.renderMain();
    this.watchdog();
    if (this.engine.currentMode === 'run') {
      // 上次用的是跑步模式：不做静置校准（引擎会拒绝），而且此刻用户很可能正跑着。
      this.showToast('已恢复跑步模式 · 日常手势保持关闭');
    } else {
      await this.calibrate();
    }
    this.loop();
  }

  /**
   * 启动后 3 秒仍收不到样本就说清楚。
   *
   * 真机上最难受的故障是"界面正常、频率显示 0 Hz、怎么甩都没反应"——
   * 用户无法判断是动作没做对还是传感器没工作。通道名一起给出来，
   * 因为他能据此选择装 APK 还是用网页版。
   */
  private watchdog(): void {
    const channel = this.motion.getStatus().channel;
    window.setTimeout(() => {
      if (this.sampleCount > 0) return;
      const hint =
        channel === 'native-sensor'
          ? '没收到传感器数据：这台设备可能没有加速度计，或系统拦截了传感器访问'
          : '没收到传感器数据：当前浏览器不支持动作传感器，建议安装 Android 安装包';
      this.showToast(hint);
      if (this.els.dot) this.els.dot.className = 'dot err';
    }, 3000);
  }

  /**
   * 静置校准：测出本机的加速度计噪声底，之后所有幅度门都按它抬高。
   * 不做这一步的话，同一套阈值在安静的小米上灵敏、在噪声大的旧机型上失灵。
   */
  private calibrate(): Promise<void> {
    return new Promise((resolve) => {
      // 跑步模式硬拒绝。UI 上已经禁用了入口，这里再拦一次是因为
      // 一个被绕过的校准会永久污染噪声底（见 engine.startCalibration 的注释）。
      if (!this.engine.startCalibration()) {
        this.showToast('跑步模式下不能校准 · 请先切回日常模式');
        resolve();
        return;
      }
      const toast = this.els['toast'];
      if (toast) this.showToast('校准中，请保持静止');
      window.setTimeout(() => {
        const floor = this.engine.finishCalibration();
        const ok = this.engine.calibrationReliable;
        this.settings.noiseFloor = floor;
        this.settings.calibrated = ok;
        store.saveSettings(this.settings);
        if (toast) {
          this.showToast(
            ok
              ? `校准完成 · 噪声底 ${floor.toFixed(2)}`
              : '校准期间检测到运动 · 已保留默认值，请静止后重试',
          );
        }
        this.renderSheet();
        resolve();
      }, 1600);
    });
  }

  // ── 主界面 ──────────────────────────────

  private renderMain(): void {
    this.root.innerHTML = '';
    const app = this.root;

    const topbar = h('div', 'topbar');
    topbar.appendChild(this.makeBrand());
    const right = h('div', 'topbar-right');
    // 模式芯片放在顶栏，因为它是全局状态而不是某一页的设置项。
    // 跑完回来忘了切回日常，是这套互斥设计下最可能出现的困惑，
    // 所以当前模式必须一直可见。
    const chip = h('div', 'mode-chip');
    chip.onclick = () => this.setMode(this.engine.currentMode === 'run' ? 'still' : 'run');
    right.appendChild(chip);
    const status = h('div', 'status');
    const dot = h('span', 'dot live');
    const hz = h('span', '', '-- Hz');
    status.append(dot, hz);
    right.appendChild(status);
    topbar.appendChild(right);
    this.els.dot = dot;
    this.els.hz = hz;
    this.els.modeChip = chip;
    this.renderModeChip();

    const stage = h('div', 'stage');
    const disc = h('div', 'disc');
    disc.appendChild(h('div', 'hole', '♪'));
    const ripple = h('div', 'ripple');
    disc.appendChild(ripple);
    stage.appendChild(disc);
    const toast = h('div', 'gesture-toast');
    stage.appendChild(toast);
    this.els.disc = disc;
    this.els.ripple = ripple;
    this.els.toast = toast;

    const now = h('div', 'now');
    const title = h('div', 'title', '未选择音乐');
    const sub = h('div', 'sub');
    const pos = h('span', '', '0:00');
    const dur = h('span', '', '0:00');
    sub.append(pos, dur);
    const progress = h('div', 'progress');
    const fill = h('div', 'fill');
    progress.appendChild(fill);
    now.append(title, sub, progress);
    this.els.title = title;
    this.els.pos = pos;
    this.els.dur = dur;
    this.els.fill = fill;

    const controls = h('div', 'controls');
    const prev = h('button', 'btn-round', '◀◀');
    prev.onclick = () => void this.player.prev();
    const mainBtn = h('button', 'btn-main', '▶');
    mainBtn.onclick = () => void this.player.toggle();
    const next = h('button', 'btn-round', '▶▶');
    next.onclick = () => void this.player.next();
    controls.append(prev, mainBtn, next);
    this.els.mainBtn = mainBtn;

    const scopeWrap = h('div', 'scope-wrap');
    const head = h('div', 'scope-head');
    const scopeTitle = h('span', '', '三轴加速度');
    head.appendChild(scopeTitle);
    const feat = h('span', '', '');
    head.appendChild(feat);
    const canvas = h('canvas');
    canvas.id = 'scope';
    scopeWrap.append(head, canvas);
    this.els.feat = feat;
    this.els.scopeTitle = scopeTitle;

    const tabs = h('div', 'tabs');
    const tabDefs: Array<[Tab, string]> = [
      ['gestures', '手势'],
      ['library', '曲库'],
      ['settings', '设置'],
    ];
    this.els.tabs = tabs;
    for (const [key, label] of tabDefs) {
      const t = h('div', 'tab' + (this.tab === key ? ' active' : ''), label);
      t.dataset.tab = key;
      tabs.appendChild(t);
    }
    tabs.onclick = (e) => {
      const target = e.target as HTMLElement;
      const key = target.dataset.tab as Tab | undefined;
      if (!key) return;
      this.tab = key;
      for (const t of Array.from(tabs.children)) {
        t.classList.toggle('active', (t as HTMLElement).dataset.tab === key);
      }
      this.renderSheet();
    };

    const sheet = h('div', 'sheet');
    this.els.sheet = sheet;

    app.append(topbar, stage, now, controls, scopeWrap, tabs, sheet);
    this.scope = new Scope(canvas);
    window.addEventListener('resize', () => this.scope?.resize());

    this.renderNow();
    this.renderSheet();
    this.paintScopeHead();
  }

  private makeBrand(): HTMLElement {
    const b = h('div', 'brand', 'SoulPlayer');
    b.appendChild(h('span', '', '手势音乐'));
    return b;
  }

  private renderNow(): void {
    const s = this.player.state;
    if (this.els.title) {
      this.els.title.textContent = s.track ? s.track.name.replace(/\.[^.]+$/, '') : '未选择音乐';
    }
    if (this.els.pos) this.els.pos.textContent = fmtTime(s.position);
    if (this.els.dur) this.els.dur.textContent = fmtTime(s.duration);
    if (this.els.fill) {
      const pct = s.duration > 0 ? (s.position / s.duration) * 100 : 0;
      this.els.fill.style.width = `${Math.min(100, pct)}%`;
    }
    if (this.els.mainBtn) this.els.mainBtn.textContent = s.playing ? '❚❚' : '▶';
    if (this.els.disc) this.els.disc.classList.toggle('playing', s.playing);
    // 曲目变了才刷新曲库页。放在 renderNow 而不是 onState 里，
    // 是因为 onState 连 timeupdate 都会触发（约 4Hz），
    // 全量重建曲库 DOM 会把列表滚动位置一起抖掉。
    const tid = s.track?.id ?? null;
    if (tid !== this.lastTrackId) {
      this.lastTrackId = tid;
      if (this.tab === 'library') this.renderSheet();
    }
  }

  // ── 主循环 ──────────────────────────────

  private loop(): void {
    cancelAnimationFrame(this.raf);
    let lastHzPaint = 0;
    const tick = (): void => {
      if (this.dirty) {
        this.dirty = false;
        const d = this.engine.evaluate();
        this.lastRunning = d.running;
        if (d.event) this.onAction(d.event);
        // 不做自动切换：检测到跑步只提示，切换由用户决定。
        // 自动切会在他正用手势时突然关掉全部常用动作，那比误触更糟。
        if (d.modeTransition === 'suggest_run') {
          this.suggested = true;
          this.showToast('检测到跑步节律 · 可在「手势」页打开运动模式');
        } else if (d.modeTransition === 'auto_paused') {
          this.showToast('已停下 · 自动暂停');
        }
        // 步频乐单匹配：只在跑步态消费 cadenceHz。滞回与驻留在匹配器里，
        // 这里只处理它发出的切换事件。
        if (this.musicMatcher && d.features) {
          const bpm = d.features.cadenceHz > 0 ? d.features.cadenceHz * 60 : null;
          const target = this.musicMatcher.update(bpm, performance.now());
          if (target) this.onMusicTarget(target);
        }
        this.scope?.draw(d.features);
        this.paintFeature(d);
      }
      const now = performance.now();
      if (now - lastHzPaint > 500) {
        lastHzPaint = now;
        const st = this.motion.getStatus();
        if (this.els.hz) this.els.hz.textContent = `${st.hz.toFixed(0)} Hz`;
        if (this.els.dot) this.els.dot.className = 'dot ' + (st.active ? 'live' : 'err');
        this.renderModeChip();
      }
      this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
  }

  private paintFeature(d: FrameDiagnostics): void {
    if (!this.els.feat) return;
    const f = d.features;
    if (!f) {
      this.els.feat.textContent = '等待数据';
      return;
    }
    if (this.engine.currentMode === 'run') {
      // 跑步时"跨度 23cm"没有任何意义（每一步都在动），要看的只有步频和稳定度：
      // 步频是不是落在跑步区间、稳定度够不够低（够低才说明是真在跑而不是手在晃）。
      const bpm = Math.round(f.cadenceHz * 60);
      this.els.feat.textContent =
        f.cadenceHz > 0
          ? `${bpm} 步/分 · 稳定 ${f.cadenceStability.toFixed(2)}`
          : '未检测到步频';
      return;
    }
    this.els.feat.textContent =
      `峰 ${f.linAccPeak.toFixed(1)} · 跨度 ${(f.lateralSpan * 100).toFixed(0)}cm · 峰数 ${f.peakCount}`;
  }

  // ── 模式 ────────────────────────────────

  /**
   * 切换使用模式。
   *
   * 两种模式互斥：切到跑步时日常六条手势整体退出匹配池，连特征都不再计算。
   * 这个"关闭"必须让用户看见——动作列表变了但没有说明的话，
   * 用户的第一反应是应用坏了，而不是模式变了。
   */
  private setMode(mode: ModeId, opts: { placement?: Placement } = {}): void {
    const prev = this.engine.currentMode;
    if (prev === mode && !opts.placement) return;
    this.engine.setMode(mode, opts.placement);
    this.suggested = false;
    this.settings.mode = mode;
    if (opts.placement) this.settings.placement = opts.placement;

    if (mode === 'run' && prev !== 'run') {
      // 互斥是绝对的（真机反馈明确要求）：打开运动模式即关掉全部日常手势开关，
      // 并持久化。不做"退出时自动恢复"——恢复旧状态会让人无法确信
      // "现在到底哪些手势是开着的"，那正是这次反馈要消除的困惑。
      for (const p of this.engine.getPresets()) {
        if (!modesOf(p).includes('still')) continue;
        this.engine.setPresetEnabled(p.id, false);
        this.settings.enabledPresets[p.id] = false;
      }
      // 步频乐单匹配器随模式创建/销毁：日常模式不消费步频，
      // 匹配器留着只会让上次跑步的判定状态泄漏到下次。
      const musicCfg = MODES.modes.run.music;
      if (musicCfg) {
        this.musicMatcher = new MusicMatcher(musicCfg);
        this.pendingSwitch = null;
        this.emptyWarned.clear();
      }
    } else if (mode !== 'run' && prev === 'run') {
      this.musicMatcher = null;
      this.pendingSwitch = null;
      this.fading = false;
      // 队列恢复全曲库。retarget 不打断正在播的歌（可能恰好是某一组的）。
      this.player.retarget(this.tracks);
    }

    store.saveSettings(this.settings);

    if (mode === 'run') {
      if (prev !== 'run') {
        this.showToast('运动模式已开启 · 日常手势已全部关闭');
        // 模式切换是全局状态变化，光靠 toast 不够——用户可能正低着头，
        // 一段不同的振动能让他确认"切过去了"。
        try {
          navigator.vibrate?.([40, 60, 40]);
        } catch {
          /* 忽略 */
        }
      }
    } else if (prev === 'run') {
      this.showToast('已退出运动模式 · 日常手势保持关闭，请在「手势」页重新开启');
      try {
        navigator.vibrate?.([40]);
      } catch {
        /* 忽略 */
      }
    }

    this.renderModeChip(true);
    this.renderSheet();
    this.paintScopeHead();
  }

  private renderModeChip(force = false): void {
    const chip = this.els.modeChip;
    if (!chip) return;
    const run = this.engine.currentMode === 'run';
    const running = run && this.lastRunning;
    // 芯片每 500ms 会被刷一次，只有状态真的变了才重建 DOM。
    const key = `${run ? (running ? 'run-active' : 'run-idle') : 'still'}${this.suggested ? '-s' : ''}`;
    if (!force && chip.dataset.key === key) return;
    chip.dataset.key = key;
    chip.className =
      'mode-chip' + (run ? ' on' : '') + (this.suggested && !run ? ' suggest' : '');
    chip.innerHTML = '';
    chip.appendChild(h('span', 'dotchip'));
    chip.appendChild(h('span', '', run ? (running ? '跑步中' : '跑步') : '日常'));
  }

  private paintScopeHead(): void {
    if (!this.els.scopeTitle) return;
    this.els.scopeTitle.textContent =
      this.engine.currentMode === 'run' ? '节律波形' : '三轴加速度';
  }

  /** 按动作区分振动节奏。跑步时不看屏幕，振动是唯一可靠的反馈通道。 */
  private vibrateFor(action: string): void {
    const map = MODES.modes[this.engine.currentMode].haptic ?? {};
    const pattern = map[action] ?? map['default'] ?? [25];
    try {
      navigator.vibrate?.(pattern);
    } catch {
      /* 桌面浏览器或不支持的机型直接忽略 */
    }
  }

  /**
   * 播报当前曲目。
   *
   * 用语音合成，但绝不阻塞：合成失败或不可用时静默退回顶部提示。
   * 不做音频 ducking——音乐正响着被压低再恢复，比不播报更烦人。
   */
  private announceTrack(): void {
    const name = this.player.state.track?.name.replace(/\.[^.]+$/, '');
    const text = name ? `当前播放 ${name}` : '还没有选择音乐';
    this.showToast(text);
    try {
      const synth = window.speechSynthesis;
      if (!synth) return;
      synth.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.lang = 'zh-CN';
      u.rate = 1.05;
      u.volume = 0.9;
      synth.speak(u);
    } catch {
      /* 无 TTS 引擎时忽略，顶部提示已经给出信息 */
    }
  }
  private onAction(ev: ActionEvent): void {
    switch (ev.action) {
      case 'player_next':
        void this.player.next();
        break;
      case 'player_prev':
        void this.player.prev();
        break;
      case 'player_toggle':
        void this.player.toggle();
        break;
      case 'player_random':
        void this.player.random();
        break;
      case 'volume_up':
        this.player.bumpVolume(0.08);
        break;
      case 'volume_down':
        this.player.bumpVolume(-0.08);
        break;
      case 'announce_track':
        this.announceTrack();
        break;
      case 'player_pause_auto':
        // 已经在暂停态就不要动，否则"暂停"会变成"暂停又播"。
        if (this.player.state.playing) this.player.pause();
        break;
      default:
        break;
    }
    // 播报自带完整文案，再叠一层"抬腕 · 播报曲目"反而是噪音
    if (ev.action !== 'announce_track') {
      this.showToast(`${ev.presetName} · ${actionLabel(ev.action)}`);
    }
    this.vibrateFor(ev.action);
    this.pulse();
    this.renderNow();
  }

  // ── 步频乐单（跑步模式）──────────────────

  private groupTracks(g: MusicGroup): Track[] {
    return this.tracks.filter((t) => (t.group ?? 'none') === g);
  }

  /** 匹配器发出目标组后的切换决策：暂停只换队列，播放按剩余时长折中。 */
  private onMusicTarget(g: MusicGroup): void {
    const list = this.groupTracks(g);
    if (list.length === 0) {
      if (!this.emptyWarned.has(g)) {
        this.emptyWarned.add(g);
        this.showToast(`「${folderLabel(g)}」文件夹还没有歌 · 继续用全曲库`);
        if (this.tab === 'gestures') this.renderSheet();
      }
      return;
    }

    const s = this.player.state;
    const label = folderLabel(g);

    // 正在播的就是目标组的歌：只把队列收敛过去，下一首自然落在组里。
    if (s.playing && (s.track?.group ?? 'none') === g) {
      this.pendingSwitch = null;
      this.player.retarget(list);
      this.showToast(`步频匹配 · 队列已切到「${label}」文件夹`);
      return;
    }

    // 时长未知（duration=0）按"剩余很长"处理——尽快匹配状态。
    const remainingMs = s.duration > 0 ? (s.duration - s.position) * 1000 : Infinity;
    const switchRemaining = MODES.modes.run.music?.switch_remaining_ms ?? 60000;

    if (!s.playing) {
      // 暂停中（多为停下自动暂停）：只换队列不起播。正在听的那首
      // 属于旧组，播完后 next() 自然进入新队列，不打断也不抢放。
      this.pendingSwitch = null;
      this.player.retarget(list);
      this.showToast(`已切到「${label}」文件夹`);
    } else if (remainingMs > switchRemaining) {
      if (this.fading) {
        this.pendingSwitch = g; // 上一段淡出还没完，等它收尾后播完再切
      } else {
        this.fadeSwapTo(g);
      }
    } else {
      this.pendingSwitch = g;
      this.showToast(`步频匹配「${label}」· 这首播完切换`);
    }
    if (this.tab === 'gestures') this.renderSheet();
  }

  /** 立即切换到目标组并（按需）起播。onEnded 兑现挂起切换时走这里。 */
  private applyRunGroup(g: MusicGroup, startPlay: boolean): void {
    const list = this.groupTracks(g);
    if (list.length === 0) return;
    this.pendingSwitch = null;
    this.player.retarget(list);
    if (startPlay) this.startFromGroup(g, list);
    else this.showToast(`已切到「${folderLabel(g)}」文件夹`);
    if (this.tab === 'gestures') this.renderSheet();
  }

  /** 0.8s 淡出 → 换队列随机起一首 → 淡入。淡出途中停下（自动暂停）则只换队列。 */
  private fadeSwapTo(g: MusicGroup): void {
    const list = this.groupTracks(g);
    if (list.length === 0) return;
    this.fading = true;
    this.pendingSwitch = null;
    const vol = this.settings.volume;
    const dur = 800;
    const t0 = performance.now();
    const down = (): void => {
      const p = (performance.now() - t0) / dur;
      // 淡出途中被暂停（典型：停下触发了自动暂停）就放弃起播，
      // 只换队列——停下的人不该被突然响起的音乐吓到。
      if (p >= 1 || !this.player.state.playing) {
        this.player.setVolume(vol);
        this.player.retarget(list);
        this.fading = false;
        if (this.player.state.playing) this.startFromGroup(g, list);
        else this.showToast(`已切到「${folderLabel(g)}」文件夹`);
        return;
      }
      this.player.setVolume(vol * (1 - p));
      requestAnimationFrame(down);
    };
    requestAnimationFrame(down);
  }

  private startFromGroup(g: MusicGroup, list: Track[]): void {
    this.showToast(`步频匹配 · 已切「${folderLabel(g)}」文件夹`);
    void this.player.playIndex(Math.floor(Math.random() * list.length)).then(() => {
      const vol = this.settings.volume;
      const t1 = performance.now();
      const up = (): void => {
        const q = (performance.now() - t1) / 600;
        this.player.setVolume(q >= 1 ? vol : vol * q);
        if (q < 1) requestAnimationFrame(up);
      };
      requestAnimationFrame(up);
    });
  }

  private showToast(text: string): void {
    const toast = this.els.toast;
    if (!toast) return;
    toast.textContent = text;
    toast.classList.add('show');
    window.clearTimeout(this.toastTimer);
    this.toastTimer = window.setTimeout(() => toast.classList.remove('show'), 1100);
  }

  private pulse(): void {
    const disc = this.els.disc;
    const ripple = this.els.ripple;
    if (disc) {
      disc.classList.remove('pulse');
      // 强制重排以重启动画
      void disc.offsetWidth;
      disc.classList.add('pulse');
    }
    if (ripple) {
      ripple.classList.remove('go');
      void ripple.offsetWidth;
      ripple.classList.add('go');
    }
  }

  // ── 面板 ────────────────────────────────

  private renderSheet(): void {
    const sheet = this.els.sheet;
    if (!sheet) return;
    sheet.innerHTML = '';
    if (this.tab === 'gestures') this.renderGestures(sheet);
    else if (this.tab === 'library') this.renderLibrary(sheet);
    else this.renderSettings(sheet);
  }

  private renderGestures(sheet: HTMLElement): void {
    const def = MODES.modes[this.engine.currentMode];
    const isRun = this.engine.currentMode === 'run';

    // 运动模式开关，放在手势开关列表首位（真机反馈要求）。
    // 它是模式级开关而不是又一条手势：打开后下方所有日常手势开关
    // 一并关闭（setMode 里做的），绝不同时开启。
    const modeRow = h('div', 'row mode-row');
    const modeLeft = h('div', 'left');
    modeLeft.appendChild(h('div', 'name', '运动模式'));
    modeLeft.appendChild(
      h(
        'div',
        'desc',
        isRun
          ? '已开启：跑步判定 + 停下自动暂停，位置档见下方'
          : '开启后日常手势全部关闭，与跑步判定互斥',
      ),
    );
    modeRow.appendChild(modeLeft);
    const modeSw = h('div', 'switch' + (isRun ? ' on' : ''));
    modeSw.onclick = () => this.setMode(isRun ? 'still' : 'run');
    modeRow.appendChild(modeSw);
    sheet.appendChild(modeRow);

    if (isRun && def.placements) {
      // 位置档决定哪些动作物理上做得出，所以放在最上面、和动作列表同屏。
      const seg = h('div', 'seg');
      for (const [key, p] of Object.entries(def.placements)) {
        if (!p) continue;
        const on = this.engine.currentPlacement === key;
        const b = h('button', on ? 'on' : '', p.label);
        b.onclick = () => this.setMode('run', { placement: key as Placement });
        seg.appendChild(b);
      }
      sheet.appendChild(seg);
      const placeNote = h(
        'div',
        'note',
        def.placements[this.engine.currentPlacement]?.note ?? '',
      );
      placeNote.style.paddingTop = '0';
      sheet.appendChild(placeNote);
    }

    if (isRun) {
      // 步频乐单状态行：让"现在放的是哪个文件夹、匹配到哪了"始终可见。
      // 匹配是自动的，用户不看屏幕时无感，回到屏幕前必须一眼能确认。
      const folders = MODES.modes.run.music?.folders ?? [];
      const counts = folders.map((f) => `${f.label} ${this.groupTracks(f.id).length}`);
      const total = folders.reduce((n, f) => n + this.groupTracks(f.id).length, 0);
      const musicRow = h('div', 'row');
      const musicLeft = h('div', 'left');
      musicLeft.appendChild(h('div', 'name', '步频乐单'));
      musicLeft.appendChild(
        h(
          'div',
          'desc',
          total === 0
            ? '三个文件夹都还没有歌——去「曲库」页放入音乐'
            : `${counts.join(' · ')} · 步频自动匹配`,
        ),
      );
      musicRow.appendChild(musicLeft);
      const target = this.musicMatcher?.target ?? null;
      const pend = this.musicMatcher?.pendingGroup ?? null;
      musicRow.appendChild(
        h('div', 'meta', target ? folderLabel(target) : pend ? `匹配中·${folderLabel(pend)}` : '待判定'),
      );
      sheet.appendChild(musicRow);
    }

    if (isRun) {
      // 互斥必须显式说出来。动作列表悄悄变了而不给解释，
      // 用户的第一反应是"应用坏了"，而不是"模式变了"。
      const banner = h('div', 'banner');
      banner.appendChild(h('b', '', '日常手势已关闭。'));
      banner.appendChild(
        document.createTextNode(
          '跑步的步频节律与甩动、摇一摇落在同一特征区间，倾斜在摆臂下也测不稳，' +
            '两者无法共存。关闭运动模式后日常手势不会自动恢复，需要逐个重新开启。' +
            '（下方的随身双击 / 抬腕属于运动模式自带动作，不受此限。）',
        ),
      );
      sheet.appendChild(banner);
    }

    const presets = this.engine.getModePresets();
    for (const p of presets) {
      const row = h('div', 'row');
      const left = h('div', 'left');
      left.appendChild(h('div', 'name', p.name));
      left.appendChild(h('div', 'desc', p.description ?? ''));
      row.appendChild(left);

      const sw = h('div', 'switch' + (p.enabled ? ' on' : ''));
      sw.onclick = () => {
        const nextOn = !sw.classList.contains('on');
        sw.classList.toggle('on', nextOn);
        this.engine.setPresetEnabled(p.id, nextOn);
        this.settings.enabledPresets[p.id] = nextOn;
        store.saveSettings(this.settings);
      };
      row.appendChild(sw);
      sheet.appendChild(row);
    }

    if (isRun) {
      sheet.appendChild(
        h(
          'div',
          'note',
          '已停用的日常手势：甩动切歌、翻面播放/暂停、摇一摇随机、倾斜调音量。' +
            '它们不只是"开关被关掉"，而是整体退出匹配池——连特征都不再计算，' +
            '所以不存在"阈值飘了误触发"的可能。上方运动模式开关关闭后，' +
            '这些手势需要逐个重新开启（不会自动恢复）。',
        ),
      );
      if (def.on_exit) {
        sheet.appendChild(
          h(
            'div',
            'note',
            `停下 ${(def.on_exit.idle_ms / 1000).toFixed(0)} 秒会自动暂停（${def.on_exit.note ?? ''}）。`,
          ),
        );
      }
      return;
    }

    const note = h(
      'div',
      'note',
      '判定链：采样 → 去重力 → 特征 → 六道门（有效/幅度/轴向/跨度/方向锁存/冷却）→ 执行。' +
        '关掉某一个手势后，它连特征计算都不再参与匹配。',
    );
    sheet.appendChild(note);
  }

  private renderLibrary(sheet: HTMLElement): void {
    const folders = MODES.modes.run.music?.folders ?? [];
    const view = this.libraryFolder;

    // 总览是根；打开文件夹或全部曲目后给返回入口。
    if (view !== null) {
      const back = h('button', 'btn ghost', '← 全部文件夹');
      back.onclick = () => {
        this.libraryFolder = null;
        this.renderSheet();
      };
      sheet.appendChild(back);
    }

    if (view === null) {
      sheet.appendChild(
        h(
          'div',
          'note',
          '三个文件夹已预设好：点「放入音乐」可一次选多首直接归入该文件夹，放错了在歌曲行尾改归属。跑步模式按步频自动在三者之间切换。',
        ),
      );
      for (const f of folders) {
        const list = this.groupTracks(f.id);
        const card = h('div', 'folder-card ' + f.id);
        const head = h('div', 'folder-head');
        head.appendChild(h('div', 'folder-name', f.label));
        head.appendChild(h('div', 'folder-count', `${list.length} 首`));
        card.appendChild(head);
        card.appendChild(h('div', 'folder-hint', f.hint ?? ''));
        const add = h('button', 'btn small', '+ 放入音乐');
        add.onclick = (e) => {
          e.stopPropagation();
          this.pickFiles(f.id);
        };
        card.appendChild(add);
        card.onclick = () => {
          this.libraryFolder = f.id;
          this.renderSheet();
        };
        sheet.appendChild(card);
      }
      const ungrouped = this.tracks.filter((t) => (t.group ?? 'none') === 'none').length;
      const allRow = h('div', 'row');
      const allLeft = h('div', 'left');
      allLeft.appendChild(h('div', 'name', '全部曲目'));
      allLeft.appendChild(
        h(
          'div',
          'desc',
          ungrouped > 0
            ? `${this.tracks.length} 首 · 其中 ${ungrouped} 首未分组`
            : `${this.tracks.length} 首`,
        ),
      );
      allRow.appendChild(allLeft);
      allRow.appendChild(h('div', 'meta', '查看 →'));
      allRow.onclick = () => {
        this.libraryFolder = 'all';
        this.renderSheet();
      };
      sheet.appendChild(allRow);
      return;
    }

    // ── 某个文件夹 / 全部曲目 ──
    const folder = folders.find((f) => f.id === view);
    const shown = view === 'all' ? this.tracks : this.groupTracks(view);

    if (folder) {
      sheet.appendChild(h('div', 'folder-title', folder.label));
      const add = h('button', 'btn accent', `+ 往「${folder.label}」放入音乐`);
      add.onclick = () => this.pickFiles(folder.id);
      sheet.appendChild(add);
      sheet.appendChild(h('div', 'note', folder.hint ?? ''));
    } else {
      const add = h('button', 'btn', '从手机中选择音乐');
      add.onclick = () => this.pickFiles('none');
      sheet.appendChild(add);
      sheet.appendChild(
        h('div', 'note', '这里能看到所有曲目。未分组的歌不参与跑步乐单，行尾下拉可把它放进某个文件夹。'),
      );
    }

    if (shown.length === 0) {
      sheet.appendChild(
        h(
          'div',
          'empty',
          folder ? `「${folder.label}」还是空的，点上面放入音乐。` : '还没有音乐，点上面选择本地音频。',
        ),
      );
      return;
    }

    const spacer = h('div');
    spacer.style.height = '8px';
    sheet.appendChild(spacer);

    for (const t of shown) {
      const isCur = this.player.state.track?.id === t.id;
      // 三重视觉提示：行高亮 + 名称前 ▶ + 尾部标签。手势切歌不看屏幕时
      // 回到曲库页，扫一眼必须能找到正在放的那首。
      const row = h('div', 'row' + (isCur ? ' cur' : ''));
      const left = h('div', 'left');
      left.appendChild(h('div', 'name', (isCur ? '▶ ' : '') + t.name.replace(/\.[^.]+$/, '')));
      left.appendChild(h('div', 'desc', `${(t.size / 1024 / 1024).toFixed(1)} MB`));
      row.appendChild(left);
      if (isCur) row.appendChild(h('span', 'tag', '播放中'));
      // 归属管理用下拉而不是标签：主流程是"选文件夹 → 放入音乐"，
      // 这里只用于纠错/移动，不要求用户逐首点。
      const sel = document.createElement('select');
      sel.className = 'group-select';
      const opts: Array<[TrackGroup, string]> = [
        ['none', '未分组'],
        ...folders.map((f) => [f.id, f.label] as [TrackGroup, string]),
      ];
      for (const [v, label] of opts) {
        const o = document.createElement('option');
        o.value = v;
        o.textContent = label;
        if ((t.group ?? 'none') === v) o.selected = true;
        sel.appendChild(o);
      }
      sel.onclick = (e) => e.stopPropagation();
      sel.onchange = () => {
        const v = sel.value as TrackGroup;
        t.group = v;
        void store.setTrackGroup(t.id, v).catch(() => undefined);
        this.renderSheet();
      };
      row.appendChild(sel);
      // 按 id 播放：跑步乐单会把队列收敛到某个文件夹，
      // 用全局下标点歌会播错（上一版曲库页的隐患）。
      row.onclick = () => void this.player.playById(t.id, this.tracks);
      sheet.appendChild(row);
    }
  }

  private renderSettings(sheet: HTMLElement): void {
    const isRun = this.engine.currentMode === 'run';
    if (isRun) {
      const banner = h('div', 'banner');
      banner.appendChild(h('b', '', '跑步模式下灵敏度与校准不可调。'));
      banner.appendChild(
        document.createTextNode(
          '跑动中测到的不是噪声底而是冲击幅度，写进去会把幅度门永久抬高，' +
            '退回日常模式后也恢复不了。跑步模式的判定用的是相对基线，本来也不需要这两项。',
        ),
      );
      sheet.appendChild(banner);
    }

    const box = h('div', 'slider-row' + (isRun ? ' disabled' : ''));
    const head = h('div', 'head');
    head.appendChild(h('span', '', '灵敏度'));
    const val = h('b', '', `${this.settings.sensitivity.toFixed(1)}×`);
    head.appendChild(val);
    const range = document.createElement('input');
    range.type = 'range';
    range.min = '0.6';
    range.max = '1.6';
    range.step = '0.05';
    range.value = String(this.settings.sensitivity);
    range.disabled = isRun;
    range.oninput = () => {
      const v = Number(range.value);
      val.textContent = `${v.toFixed(1)}×`;
      this.engine.setSensitivity(v);
      this.settings.sensitivity = v;
      store.saveSettings(this.settings);
    };
    box.append(head, range);
    sheet.appendChild(box);

    const info = h('div', 'row');
    const left = h('div', 'left');
    left.appendChild(h('div', 'name', '本机噪声底'));
    left.appendChild(h('div', 'desc', '启动校准测得，幅度门按它自动抬高'));
    info.appendChild(left);
    info.appendChild(h('div', 'meta', this.settings.noiseFloor.toFixed(2)));
    sheet.appendChild(info);

    // 授权"没反应"的反馈就是这里缺一行状态：步检测授权后静默工作，
    // 用户没有任何地方能确认它到底启没启用。
    const stepRow = h('div', 'row');
    const stepLeft = h('div', 'left');
    stepLeft.appendChild(h('div', 'name', '硬件步检测'));
    stepLeft.appendChild(h('div', 'desc', '跑步提示的第二确认源，Android 10+ 需授权'));
    stepRow.appendChild(stepLeft);
    stepRow.appendChild(h('div', 'meta', this.stepDetectorOn ? '已启用' : '未启用'));
    sheet.appendChild(stepRow);

    const recal = h(
      'button',
      'btn',
      isRun ? '跑步模式下不可校准' : '重新校准（保持静止 1.6 秒）',
    );
    recal.disabled = isRun;
    recal.onclick = () => {
      if (isRun) return;
      void this.calibrate();
    };
    sheet.appendChild(recal);

    const volRow = h('div', 'slider-row');
    const vh = h('div', 'head');
    vh.appendChild(h('span', '', '音量'));
    const vv = h('b', '', `${Math.round(this.settings.volume * 100)}%`);
    vh.appendChild(vv);
    const vr = document.createElement('input');
    vr.type = 'range';
    vr.min = '0';
    vr.max = '1';
    vr.step = '0.02';
    vr.value = String(this.settings.volume);
    vr.oninput = () => {
      const v = Number(vr.value);
      vv.textContent = `${Math.round(v * 100)}%`;
      this.player.setVolume(v);
      this.settings.volume = v;
      store.saveSettings(this.settings);
    };
    volRow.append(vh, vr);
    sheet.appendChild(volRow);

    const clearBtn = h('button', 'btn ghost', '清空曲库');
    clearBtn.onclick = async () => {
      await store.clearTracks();
      this.tracks = [];
      this.player.setTracks([]);
      this.libraryFolder = null;
      this.renderSheet();
      this.renderNow();
    };
    sheet.appendChild(clearBtn);

    const note = h(
      'div',
      'note',
      '识别完全在本机完成：传感器数据不进网络，曲目文件也只存在浏览器本地存储里。' +
        '判定阈值来自 contracts/sensor_presets.json，改那里即可，代码里没有第二份。',
    );
    sheet.appendChild(note);
  }

  // ── 曲库 ────────────────────────────────

  /** 选一批本地音频并归入指定文件夹（'none' = 未分组）。 */
  private pickFiles(group: TrackGroup = 'none'): void {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'audio/*';
    input.multiple = true;
    input.onchange = async () => {
      const files = input.files;
      if (!files) return;
      for (const f of Array.from(files)) {
        if (f.size > 60 * 1024 * 1024) continue; // 单曲 60MB 上限，避免占满配额
        await store.addTrack(f, group).catch(() => undefined);
      }
      this.tracks = await store.listTracks();
      await this.attachTrackUrls();
      this.player.setTracks(this.tracks);
      // 导入后直接停在目标文件夹里，用户立刻能看到刚放进去的歌。
      // 未分组的导入跳到「全部曲目」，不会掉回文件夹总览。
      if (group !== 'none') this.libraryFolder = group;
      else if (this.libraryFolder === null) this.libraryFolder = 'all';
      // 跑步态下已锁定目标文件夹的话，队列仍收敛到该文件夹。
      const target = this.musicMatcher?.target;
      if (target) {
        const list = this.groupTracks(target);
        if (list.length > 0) this.player.retarget(list);
      }
      this.renderSheet();
      this.renderNow();
    };
    input.click();
  }

  /** 把 IndexedDB 里的 Blob 换成本次的 ObjectURL，播放器才吃得到。 */
  private async attachTrackUrls(): Promise<void> {
    for (const t of this.tracks) {
      const blob = await store.getTrackBlob(t.id).catch(() => null);
      if (blob) this.player.registerUrl(t.id, URL.createObjectURL(blob));
    }
  }
}
