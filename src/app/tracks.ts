import type { StoredTrack, TrackSettings } from './track-store';

export type TrackStatus = 'generating' | 'ready' | 'canceled' | 'error';

export type Track = {
  id: string;
  createdAt: number;
  status: TrackStatus;
  actualSeconds: number;
  settings: TrackSettings;
  wav?: Blob;
  error?: string;
};

export const INSTRUMENTAL_LYRICS = '[instrumental]';

export function combineFinePrompt(settings: {
  fineMeta: string;
  fineVocal: string;
  fineArrangement: string;
}): string {
  return [settings.fineMeta, settings.fineVocal, settings.fineArrangement]
    .map((part) => part.trim())
    .filter(Boolean)
    .join('\n\n');
}

export function promptFromSettings(settings: TrackSettings): string {
  if (settings.mode === 'raw' && settings.rawPrompt.trim()) return settings.rawPrompt.trim();
  return combineFinePrompt(settings);
}

export function lyricsFromSettings(settings: TrackSettings): string {
  return settings.instrumental ? INSTRUMENTAL_LYRICS : settings.lyrics.trim();
}

export function trackFromStored(stored: StoredTrack): Track {
  return {
    id: stored.id,
    createdAt: stored.createdAt,
    status: 'ready',
    actualSeconds: stored.actualSeconds,
    settings: stored.settings,
    wav: stored.wav,
  };
}

export function trackToStored(track: Track, wav: Blob): StoredTrack {
  return {
    id: track.id,
    createdAt: track.createdAt,
    actualSeconds: track.actualSeconds,
    settings: track.settings,
    wav,
  };
}

export function randomSeed(): number {
  return Math.floor(Math.random() * 4_294_967_296);
}
