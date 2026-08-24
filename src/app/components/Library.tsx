import type { MouseEvent } from 'react';
import { formatClock, formatTimeOfDay } from '../format';
import type { Messages } from '../i18n';
import type { Track } from '../tracks';
import { EqBars, PauseIcon, PlayIcon } from './icons';

export type LibraryProps = {
  tr: Messages;
  tracks: Track[];
  selectedId: string | null;
  currentId: string | null;
  playing: boolean;
  generationPercent: number;
  onSelect: (track: Track) => void;
  onPlayToggle: (track: Track) => void;
  onDelete: (track: Track) => void;
  onNew: () => void;
};

function subLine(tr: Messages, track: Track, generationPercent: number) {
  if (track.status === 'generating') return { text: `${tr.stGen} · ${generationPercent}%`, tone: 'text-accent' };
  if (track.status === 'canceled') return { text: tr.stCanceled, tone: 'text-danger' };
  if (track.status === 'error') return { text: tr.stError, tone: 'text-danger' };
  return {
    text: `${formatClock(track.actualSeconds)} · Seed ${track.settings.seed}`,
    tone: 'text-muted',
  };
}

export function Library({
  tr,
  tracks,
  selectedId,
  currentId,
  playing,
  generationPercent,
  onSelect,
  onPlayToggle,
  onDelete,
  onNew,
}: LibraryProps) {
  return (
    <>
      <div className="sticky top-0 z-[2] flex items-center gap-2 border-b border-line bg-panel px-3.5 py-3">
        <div className="text-base font-bold">{tr.histTitle}</div>
        <div className="font-mono text-xs text-muted2">
          {tracks.length}
          {tr.tracksSuffix}
        </div>
        <div className="flex-1" />
        <button
          type="button"
          onClick={onNew}
          className="rounded-[7px] border border-line bg-accent-soft px-2.5 py-1 text-sm font-bold text-accent hover:border-accent/40 hover:bg-accent/25"
        >
          {tr.newBtn}
        </button>
      </div>
      <div className="flex flex-col gap-1 p-2.5">
        {tracks.map((track) => {
          const selected = track.id === selectedId;
          const isCurrent = track.id === currentId;
          const nowPlaying = isCurrent && playing;
          const sub = subLine(tr, track, generationPercent);
          const stopThen = (handler: () => void) => (event: MouseEvent) => {
            event.stopPropagation();
            handler();
          };
          return (
            <div
              key={track.id}
              onClick={() => onSelect(track)}
              className={`flex cursor-pointer items-center gap-2.5 rounded-[10px] border p-2.5 transition-colors ${
                selected ? 'border-line bg-panel2' : 'border-transparent bg-transparent hover:bg-panel2/60'
              }`}
            >
              <button
                type="button"
                onClick={stopThen(() => onPlayToggle(track))}
                disabled={track.status !== 'ready'}
                aria-label={nowPlaying ? tr.pause : tr.play}
                className={`flex size-[30px] flex-none items-center justify-center rounded-full border border-line p-0 disabled:opacity-45 ${
                  isCurrent
                    ? 'bg-accent text-accent-fg enabled:hover:bg-accent2'
                    : 'bg-transparent text-muted enabled:hover:border-muted2 enabled:hover:text-ink'
                }`}
              >
                {nowPlaying ? <PauseIcon size={10} /> : <PlayIcon size={10} />}
              </button>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-1.5">
                  <div className="overflow-hidden text-ellipsis whitespace-nowrap text-sm font-semibold">
                    {track.settings.title}
                  </div>
                  {nowPlaying && <EqBars />}
                </div>
                <div
                  className={`mt-[3px] overflow-hidden text-ellipsis whitespace-nowrap font-mono text-xs ${sub.tone}`}
                >
                  {sub.text}
                </div>
              </div>
              <div className="flex flex-none flex-col items-end gap-1">
                <div className="font-mono text-xs text-muted2">{formatTimeOfDay(track.createdAt)}</div>
                <button
                  type="button"
                  onClick={stopThen(() => onDelete(track))}
                  title={tr.delTitle}
                  className="border-none bg-transparent px-1 py-0.5 text-base leading-none text-muted2 hover:text-danger"
                >
                  ×
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </>
  );
}
