/**
 * 播放器适配层。
 *
 * 用原生 <audio> + Media Session API，而不是 Web Audio 的 AudioContext：
 * 前者能直接吃 Blob URL、能在 Android 上后台播放并接管锁屏/耳机按键，
 * 后者要自己写解码和调度，收益不成正比。
 *
 * 手势驱动的一个特殊要求：动作本身不是"用户手势"，浏览器不允许在
 * 非手势上下文中调用 play()。所以首次播放必须由 UI 按钮解锁一次，
 * 之后手势触发的 play() 才能生效。这个坑不处理会在真机上表现为
 * "识别到了但没声音"。
 */

/**
 * 曲目分组。跑步模式按步频在三个预设文件夹之间切换乐单，
 * 'none' = 未分组（不参与跑步乐单匹配，日常模式照常播放）。
 * 类型真源在 core/types.ts（契约字段一一对应），这里 re-export 让
 * platform 层的调用方不用跨层 import。
 */
import type { TrackGroup } from '../core/types';
export type { TrackGroup };

export interface Track {
  id: string;
  name: string;
  size: number;
  type: string;
  addedAt: number;
  /** 缺省视为 'none'——老数据没有这个字段，不迁移也不丢。 */
  group?: TrackGroup;
}

export interface PlayerState {
  playing: boolean;
  index: number;
  track: Track | null;
  position: number;
  duration: number;
  volume: number;
  count: number;
}

export class Player {
  private audio = new Audio();
  private tracks: Track[] = [];
  private urls = new Map<string, string>();
  private index = -1;
  private volume = 0.9;
  /** 首次用户手势解锁前，play() 会被浏览器拒绝。 */
  private unlocked = false;

  onState: (() => void) | null = null;
  onEnded: (() => void) | null = null;

  constructor() {
    this.audio.preload = 'auto';
    this.audio.addEventListener('timeupdate', () => this.onState?.());
    this.audio.addEventListener('play', () => {
      this.syncMediaSession();
      this.onState?.();
    });
    this.audio.addEventListener('pause', () => this.onState?.());
    this.audio.addEventListener('ended', () => {
      this.onEnded?.();
      this.onState?.();
    });
    this.audio.addEventListener('loadedmetadata', () => this.onState?.());
    this.setupMediaSession();
  }

  setTracks(tracks: Track[]): void {
    this.tracks = tracks;
    if (this.index >= tracks.length) this.index = tracks.length - 1;
    this.onState?.();
  }

  /**
   * 跑步乐单切换用的队列重定向：换成新列表但不打断正在播的歌。
   *
   * 与 setTracks 的区别：当前曲在新列表里时，索引跟着平移到它的新位置
   * （next() 从它后面继续）；不在时保持播放不动，索引夹到合法范围，
   * 下一首自然落进新列表。绝不重设 audio.src——正在听的歌突然断掉，
   * 比"晚一首歌才匹配上节奏"糟糕得多。
   */
  retarget(tracks: Track[]): void {
    const curId = this.index >= 0 ? this.tracks[this.index]?.id : undefined;
    this.tracks = tracks;
    if (curId) {
      const i = tracks.findIndex((t) => t.id === curId);
      if (i >= 0) this.index = i;
    }
    if (this.index >= tracks.length) this.index = tracks.length - 1;
    this.onState?.();
  }

  /** 由 platform/storage 在载入曲目时注册 Blob URL。 */
  registerUrl(id: string, url: string): void {
    this.urls.set(id, url);
  }

  get state(): PlayerState {
    const track = this.index >= 0 ? this.tracks[this.index] ?? null : null;
    return {
      playing: !this.audio.paused && !this.audio.ended,
      index: this.index,
      track,
      position: this.audio.currentTime || 0,
      duration: this.audio.duration || 0,
      volume: this.volume,
      count: this.tracks.length,
    };
  }

  /**
   * 必须在用户手势里调一次，让浏览器放行后续的程序化播放。
   *
   * 关键教训：**绝不能 await 无源的 play()**。实测桌面 Edge 上对一个没有
   * src 的 Audio 调 play()，返回的 promise 永远 pending（既不 resolve 也不
   * reject），await 它会把整个启动流程吊死在"正在请求权限…"。
   * 解锁的意义在于"在手势上下文里触发过一次"，结果无关紧要，所以
   * fire-and-forget，用定时器收尾静音状态。
   */
  unlock(): void {
    if (this.unlocked) return;
    this.audio.muted = true;
    void this.audio.play().catch(() => undefined);
    window.setTimeout(() => {
      this.audio.pause();
      this.audio.muted = false;
      this.unlocked = true;
    }, 150);
  }

  async playIndex(i: number): Promise<void> {
    if (this.tracks.length === 0) return;
    const idx = ((i % this.tracks.length) + this.tracks.length) % this.tracks.length;
    const track = this.tracks[idx];
    const url = this.urls.get(track.id);
    if (!url) return;
    if (this.index !== idx || this.audio.src !== url) {
      this.index = idx;
      this.audio.src = url;
    }
    this.audio.volume = this.volume;
    try {
      await this.audio.play();
      this.unlocked = true;
    } catch {
      /* 未解锁时静默失败，UI 会显示为"未播放" */
    }
    this.onState?.();
  }

  /**
   * 按曲目 id 播放。曲库页在跑步态下队列可能已被收敛到某个文件夹，
   * 用全局下标点歌会播错；这里按 id 找，找不到（该曲不在当前队列）
   * 时把队列扩回全曲库再播——手动选曲是显式意图，优先于自动乐单，
   * 下次步频变档时 app 层会重新把队列收敛回去。
   */
  async playById(id: string, allTracks?: Track[]): Promise<void> {
    let i = this.tracks.findIndex((t) => t.id === id);
    if (i < 0 && allTracks) {
      this.setTracks(allTracks);
      i = this.tracks.findIndex((t) => t.id === id);
    }
    if (i >= 0) await this.playIndex(i);
  }

  async play(): Promise<void> {
    if (this.index < 0 && this.tracks.length > 0) {
      await this.playIndex(0);
      return;
    }
    try {
      await this.audio.play();
      this.unlocked = true;
    } catch {
      /* 同上 */
    }
    this.onState?.();
  }

  pause(): void {
    this.audio.pause();
  }

  /** 翻转手势只关心"切换"，不区分当前是播放还是暂停。 */
  async toggle(): Promise<boolean> {
    if (this.audio.paused) {
      if (this.index < 0 && this.tracks.length > 0) await this.playIndex(0);
      else await this.play();
      return true;
    }
    this.pause();
    return false;
  }

  async next(): Promise<void> {
    if (this.tracks.length === 0) return;
    await this.playIndex(this.index < 0 ? 0 : this.index + 1);
  }

  async prev(): Promise<void> {
    if (this.tracks.length === 0) return;
    // 播放超过 3 秒时，"上一首"先回到本曲开头，与主流播放器一致
    if (this.audio.currentTime > 3) {
      this.audio.currentTime = 0;
      return;
    }
    await this.playIndex(this.index <= 0 ? this.tracks.length - 1 : this.index - 1);
  }

  async random(): Promise<void> {
    if (this.tracks.length === 0) return;
    if (this.tracks.length === 1) {
      await this.playIndex(0);
      return;
    }
    let next = this.index;
    while (next === this.index) next = Math.floor(Math.random() * this.tracks.length);
    await this.playIndex(next);
  }

  setVolume(v: number): void {
    this.volume = Math.min(1, Math.max(0, v));
    this.audio.volume = this.volume;
    this.onState?.();
  }

  /** 音量步进 8%，与按住倾斜的重复频率（约 5.5 次/秒）配合，1.5 秒可从 0 到满。 */
  bumpVolume(delta: number): void {
    this.setVolume(this.volume + delta);
  }

  seekRatio(r: number): void {
    if (!Number.isFinite(this.audio.duration)) return;
    this.audio.currentTime = Math.min(1, Math.max(0, r)) * this.audio.duration;
    this.onState?.();
  }

  private syncMediaSession(): void {
    if (!('mediaSession' in navigator)) return;
    const t = this.state.track;
    if (!t) return;
    navigator.mediaSession.metadata = new MediaMetadata({
      title: t.name.replace(/\.[^.]+$/, ''),
      artist: 'SoulPlayer · 手势控制',
      album: 'SoulPlayer Mobile',
    });
    navigator.mediaSession.playbackState = this.audio.paused ? 'paused' : 'playing';
  }

  private setupMediaSession(): void {
    if (!('mediaSession' in navigator)) return;
    const ms = navigator.mediaSession;
    const bind = (action: MediaSessionAction, fn: () => void): void => {
      try {
        ms.setActionHandler(action, fn);
      } catch {
        /* 部分动作在旧浏览器不存在，忽略 */
      }
    };
    bind('play', () => void this.play());
    bind('pause', () => this.pause());
    bind('previoustrack', () => void this.prev());
    bind('nexttrack', () => void this.next());
  }
}
