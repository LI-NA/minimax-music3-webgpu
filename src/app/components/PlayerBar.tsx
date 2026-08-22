import type { MouseEvent } from 'react';
import { formatClock } from '../format';
import type { Messages } from '../i18n';
import type { Track } from '../tracks';
import { NextIcon, PauseIcon, PlayIcon, PrevIcon } from './icons';

export type PlayerBarProps = {
  tr: Messages;
  track: Track | null;
  playing: boolean;
  position: number;
  volume: number;
  loop: boolean;
  isMobile: boolean;
  hasReady: boolean;
  onPrev: () => void;
  onNext: () => void;
  onTogglePlay: () => void;
  onSeek: (fraction: number) => void;
  onVolume: (volume: number) => void;
  onToggleLoop: () => void;
  onDownload: () => void;
};

export function PlayerBar({
  tr,
  track,
  playing,
  position,
  volume,
  loop,
  isMobile,
  hasReady,
  onPrev,
  onNext,
  onTogglePlay,
  onSeek,
  onVolume,
  onToggleLoop,
  onDownload,
}: PlayerBarProps) {
  const total = track?.actualSeconds ?? 0;
  const fraction = total > 0 ? Math.min(1, position / total) : 0;

  const handleSeek = (event: MouseEvent<HTMLDivElement>) => {
    if (!track) return;
    const rect = event.currentTarget.getBoundingClientRect();
    onSeek(Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)));
  };

  return (
    <footer className="flex h-[72px] flex-none items-center gap-3.5 border-t border-line bg-panel px-4">
      <div className="flex items-center gap-1.5">
        <button
          type="button"
          onClick={onPrev}
          disabled={!hasReady}
          aria-label="previous"
          className="flex size-8 items-center justify-center rounded-full border-none bg-transparent p-0 text-muted disabled:opacity-45"
        >
          <PrevIcon size={13} />
        </button>
        <button
          type="button"
          onClick={onTogglePlay}
          disabled={!track}
          aria-label={playing ? tr.pause : tr.play}
          className="flex size-[42px] items-center justify-center rounded-full border-none bg-accent p-0 text-accent-fg disabled:opacity-45"
        >
          {playing ? <PauseIcon size={15} /> : <PlayIcon size={15} />}
        </button>
        <button
          type="button"
          onClick={onNext}
          disabled={!hasReady}
          aria-label="next"
          className="flex size-8 items-center justify-center rounded-full border-none bg-transparent p-0 text-muted disabled:opacity-45"
        >
          <NextIcon size={13} />
        </button>
      </div>
      <div className="w-[150px] min-w-0">
        <div className="overflow-hidden text-ellipsis whitespace-nowrap text-[12.5px] font-semibold">
          {track ? track.settings.title : tr.npNone}
        </div>
        <div className="mt-0.5 overflow-hidden text-ellipsis whitespace-nowrap font-mono text-[9.5px] text-muted2">
          {track ? `Seed ${track.settings.seed} · WAV 44.1 kHz` : ''}
        </div>
      </div>
      <div className="flex min-w-0 flex-1 items-center gap-2.5">
        <div className="flex-none font-mono text-[10.5px] text-muted">{formatClock(Math.min(position, total))}</div>
        <div onClick={handleSeek} className="flex h-5 flex-1 cursor-pointer items-center">
          <div className="h-1 flex-1 overflow-hidden rounded-sm bg-panel2">
            <div className="h-full rounded-sm bg-accent" style={{ width: `${fraction * 100}%` }} />
          </div>
        </div>
        <div className="flex-none font-mono text-[10.5px] text-muted">{formatClock(total)}</div>
      </div>
      {!isMobile && (
        <div className="flex items-center gap-2.5">
          <div className="font-mono text-[9.5px] text-muted2">VOL</div>
          <input
            type="range"
            min={0}
            max={100}
            value={volume}
            onChange={(event) => onVolume(Number(event.target.value))}
            className="w-[84px]"
            aria-label="volume"
          />
          <button
            type="button"
            onClick={onToggleLoop}
            className={`rounded-[7px] border px-2.5 py-[5px] font-mono text-[10px] ${
              loop ? 'border-accent bg-accent-soft text-accent' : 'border-line bg-transparent text-muted2'
            }`}
          >
            LOOP
          </button>
          <button
            type="button"
            onClick={onDownload}
            disabled={!track || track.status !== 'ready'}
            className="rounded-[7px] border border-line bg-transparent px-2.5 py-[5px] font-mono text-[10px] text-muted disabled:opacity-45"
          >
            WAV
          </button>
        </div>
      )}
    </footer>
  );
}
