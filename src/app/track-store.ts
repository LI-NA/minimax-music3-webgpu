import type { MusicSamplingInput } from '../workers/protocol';

export type PromptMode = 'fine' | 'raw';

export type TrackSettings = {
  title: string;
  mode: PromptMode;
  fineMeta: string;
  fineVocal: string;
  fineArrangement: string;
  rawPrompt: string;
  lyrics: string;
  instrumental: boolean;
  durationSeconds: number;
  seed: number;
  sampling: MusicSamplingInput;
};

export type StoredTrack = {
  id: string;
  createdAt: number;
  actualSeconds: number;
  settings: TrackSettings;
  wav: Blob;
};

const DB_NAME = 'minimax-music3';
const DB_VERSION = 1;
const STORE = 'tracks';

function requestPromise<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('IndexedDB request failed'));
  });
}

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(STORE)) request.result.createObjectStore(STORE, { keyPath: 'id' });
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('IndexedDB open failed'));
  });
}

async function withStore<T>(mode: IDBTransactionMode, run: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const database = await openDatabase();
  try {
    return await requestPromise(run(database.transaction(STORE, mode).objectStore(STORE)));
  } finally {
    database.close();
  }
}

export async function listStoredTracks(): Promise<StoredTrack[]> {
  const tracks = await withStore('readonly', (store) => store.getAll() as IDBRequest<StoredTrack[]>);
  return tracks.sort((left, right) => right.createdAt - left.createdAt);
}

export async function saveStoredTrack(track: StoredTrack): Promise<void> {
  await withStore('readwrite', (store) => store.put(track));
}

export async function deleteStoredTrack(id: string): Promise<void> {
  await withStore('readwrite', (store) => store.delete(id));
}
