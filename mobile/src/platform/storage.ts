import type { Track, TrackGroup } from './player';
import type { ModeId, Placement } from '../core/types';

/**
 * 本地持久化。分两个 IndexedDB store：
 *   tracks —— 只有元数据，列表页一次全读，几十条也就几 KB
 *   blobs  —— 音频二进制，只在真正播放某首时才单独取
 *
 * 为什么不合成一个 store：合成的话 getAll() 会把所有音频的 Blob 一起读进内存，
 * 导入 20 首歌就是上百 MB，手机浏览器会直接被杀掉。
 */

const DB_NAME = 'soulplayer-mobile';
const DB_VERSION = 1;
const STORE_TRACKS = 'tracks';
const STORE_BLOBS = 'blobs';
const SETTINGS_KEY = 'soulplayer.settings.v1';

export interface Settings {
  sensitivity: number;
  noiseFloor: number;
  enabledPresets: Record<string, boolean>;
  volume: number;
  calibrated: boolean;
  /** 上次使用的模式。重启后沿用，跑步的人不想每次重新点一次。 */
  mode: ModeId;
  /** 手机在跑动中的位置。决定跑步模式下哪些手势可用。 */
  placement: Placement;
}

export const DEFAULT_SETTINGS: Settings = {
  sensitivity: 1,
  noiseFloor: 0.6,
  enabledPresets: {},
  volume: 0.9,
  calibrated: false,
  mode: 'still',
  placement: 'hand',
};

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_TRACKS)) {
        db.createObjectStore(STORE_TRACKS, { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains(STORE_BLOBS)) {
        db.createObjectStore(STORE_BLOBS);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function tx<T>(store: string, mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return openDb().then(
    (db) =>
      new Promise<T>((resolve, reject) => {
        const t = db.transaction(store, mode);
        const req = fn(t.objectStore(store));
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
      }),
  );
}

/** 导入一首歌并归入指定文件夹（缺省 'none' = 未分组）。 */
export async function addTrack(file: File, group: TrackGroup = 'none'): Promise<Track> {
  const track: Track = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
    name: file.name,
    size: file.size,
    type: file.type,
    addedAt: Date.now(),
    group,
  };
  await tx(STORE_BLOBS, 'readwrite', (s) => s.put(file, track.id));
  await tx(STORE_TRACKS, 'readwrite', (s) => s.put(track));
  return track;
}

export async function listTracks(): Promise<Track[]> {
  const all = await tx<Track[]>(STORE_TRACKS, 'readonly', (s) => s.getAll() as IDBRequest<Track[]>);
  return all.sort((a, b) => a.addedAt - b.addedAt);
}

export async function getTrackBlob(id: string): Promise<Blob | null> {
  const blob = await tx<Blob | undefined>(STORE_BLOBS, 'readonly', (s) => s.get(id) as IDBRequest<Blob | undefined>);
  return blob ?? null;
}

export async function removeTrack(id: string): Promise<void> {
  await tx(STORE_BLOBS, 'readwrite', (s) => s.delete(id));
  await tx(STORE_TRACKS, 'readwrite', (s) => s.delete(id));
}

/**
 * 改曲目所属的预设文件夹（舒缓/轻松/激烈/未分组）。
 *
 * group 只是 tracks store 里的一条普通字段，不需要升 DB 版本建索引：
 * 分组只有读整个列表时才用到（几十条全量过滤），索引收益为零。
 */
export async function setTrackGroup(id: string, group: TrackGroup): Promise<void> {
  const t = await tx<Track | undefined>(
    STORE_TRACKS,
    'readonly',
    (s) => s.get(id) as IDBRequest<Track | undefined>,
  );
  if (!t) return;
  t.group = group;
  await tx(STORE_TRACKS, 'readwrite', (s) => s.put(t));
}

export async function clearTracks(): Promise<void> {
  await tx(STORE_TRACKS, 'readwrite', (s) => s.clear());
  await tx(STORE_BLOBS, 'readwrite', (s) => s.clear());
}

export function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) return { ...DEFAULT_SETTINGS };
    const s = { ...DEFAULT_SETTINGS, ...(JSON.parse(raw) as Partial<Settings>) };
    // 旧版本可能在跑动中校准过，把噪声底写成了接近上限的值。那个值会让
    // stillGate 变成 9，加速度永远落不进"静止"，速度衰减永不触发，
    // 位移无界漂移——而且它是持久化的，用户只能卸载重装才能恢复。
    // 正常的静置校准不会超过 1.5，超过 2.0 一律视为被污染。
    if (s.noiseFloor > 2.0) {
      s.noiseFloor = DEFAULT_SETTINGS.noiseFloor;
      s.calibrated = false;
    }
    return s;
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

export function saveSettings(s: Settings): void {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
  } catch {
    /* 隐私模式下 localStorage 可能不可写，静默降级 */
  }
}
