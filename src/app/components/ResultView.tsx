import type { MouseEvent } from 'react';
import { artifactErrorMessage } from '../artifact-cache-ui';
import { formatClock, formatTimeOfDay } from '../format';
import type { Messages } from '../i18n';
import { lyricsFromSettings, promptFromSettings, type Track } from '../tracks';
import { PauseIcon, PlayIcon } from './icons';

const actionButton =
  'rounded-[9px] border border-line bg-panel px-4 py-[9px] text-sm font-semibold text-ink ' +
  'disabled:cursor-not-allowed disabled:opacity-45';

export type ResultViewProps = {
  tr: Messages;
  track: Track;
  bars: number[];
  isCurrent: boolean;
  playing: boolean;
  posFraction: number;
  canVariation: boolean;
  onWaveClick: (fraction: number) => void;
  onTogglePlay: () => void;
  onDownload: () => void;
  onReuse: () => void;
  onVariation: () => void;
  onDelete: () => void;
};

export function ResultView({
  tr,
  track,
  bars,
  isCurrent,
  playing,
  posFraction,
  canVariation,
  onWaveClick,
  onTogglePlay,
  onDownload,
  onReuse,
  onVariation,
  onDelete,
}: ResultViewProps) {
  const ready = track.status === 'ready';
  const canceled = track.status === 'canceled';
  const failed = track.status === 'error';
  const statusLabel = ready ? tr.stReady : canceled ? tr.stCanceled : failed ? tr.stError : tr.stGen;
  const selPlaying = isCurrent && playing;
  const settings = track.settings;
  const settingsText = [
    `flow guidance   ${settings.sampling.flowGuidance}    flow steps   ${settings.sampling.flowSteps}    global guidance   ${settings.sampling.globalGuidance}`,
    `semantic top-k  ${settings.sampling.semanticTopK}    residual top-k  ${settings.sampling.residualTopK}    temperature   ${settings.sampling.temperature}`,
  ].join('\n');
  const metaCells = [
    { key: tr.kLen, value: ready ? formatClock(track.actualSeconds) : '--' },
    { key: tr.kReq, value: `${settings.durationSeconds}s` },
    { key: tr.kSeed, value: String(settings.seed) },
    { key: tr.kMode, value: settings.mode === 'fine' ? tr.fineTab : 'Raw' },
    { key: tr.kVocal, value: settings.instrumental ? tr.inst : tr.vocal },
    { key: tr.kTime, value: formatTimeOfDay(track.createdAt) },
  ];

  const handleWaveClick = (event: MouseEvent<HTMLDivElement>) => {
    if (!ready) return;
    const rect = event.currentTarget.getBoundingClientRect();
    onWaveClick(Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)));
  };

  return (
    <div className="mx-auto flex max-w-[640px] flex-col gap-[18px]">
      <div className="flex items-start gap-3">
        <div className="min-w-0 flex-1">
          <div className="font-mono text-xs tracking-[.12em] text-muted">{tr.resultLabel}</div>
          <div className="mt-1 text-2xl font-bold tracking-tight">{settings.title}</div>
          <div className="mt-1.5 font-mono text-xs text-muted">
            Seed {settings.seed} · {settings.mode === 'fine' ? tr.fineTab : tr.rawTab} ·{' '}
            {settings.instrumental ? tr.inst : tr.vocal}
          </div>
        </div>
        <div
          className={`mt-1 whitespace-nowrap rounded-full px-2.5 py-1 font-mono text-xs tracking-[.08em] ${
            canceled || failed ? 'bg-danger/15 text-danger' : 'bg-accent-soft text-accent'
          }`}
        >
          {statusLabel}
        </div>
      </div>

      {failed && (
        <div role="alert" className="text-sm leading-normal text-danger">
          {artifactErrorMessage(tr, track.errorCode, tr.errGeneration)}
          {track.error && <div className="mt-1 font-mono text-xs leading-normal text-muted2">{track.error}</div>}
        </div>
      )}

      <div
        onClick={handleWaveClick}
        className={`flex h-28 items-center gap-[2px] rounded-xl border border-line bg-panel px-4 py-[18px] ${
          ready ? 'cursor-pointer' : ''
        }`}
      >
        {bars.map((height, index) => (
          <div
            key={index}
            className={`flex-1 rounded-[1.5px] ${
              isCurrent && index / bars.length < posFraction ? 'bg-accent' : 'bg-wave'
            }`}
            style={{ height: `${height}%` }}
          />
        ))}
      </div>

      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={onTogglePlay}
          disabled={!ready}
          className="flex items-center gap-2 rounded-[9px] border-none bg-accent px-5 py-[9px] text-sm font-bold text-accent-fg disabled:cursor-not-allowed disabled:opacity-45"
        >
          {selPlaying ? <PauseIcon size={11} /> : <PlayIcon size={11} />}
          {selPlaying ? tr.pause : tr.play}
        </button>
        <button type="button" onClick={onDownload} disabled={!ready} className={actionButton}>
          {tr.dlWav}
        </button>
        <button type="button" onClick={onReuse} className={actionButton}>
          {tr.reuse}
        </button>
        <button type="button" onClick={onVariation} disabled={!canVariation} className={actionButton}>
          {tr.variation}
        </button>
        <button
          type="button"
          onClick={onDelete}
          className="rounded-[9px] border border-line bg-transparent px-4 py-[9px] text-sm font-semibold text-danger"
        >
          {tr.del}
        </button>
      </div>

      <div className="grid grid-cols-[repeat(auto-fit,minmax(150px,1fr))] gap-2">
        {metaCells.map((cell) => (
          <div key={cell.key} className="rounded-[10px] border border-line bg-panel px-3 py-2.5">
            <div className="font-mono text-xs uppercase tracking-[.1em] text-muted2">{cell.key}</div>
            <div className="mt-[5px] font-mono text-base text-ink">{cell.value}</div>
          </div>
        ))}
      </div>

      <div>
        <div className="mb-2 font-mono text-xs tracking-[.12em] text-muted">{tr.modelSettings}</div>
        <div className="whitespace-pre-wrap rounded-[10px] border border-line bg-input px-4 py-3 font-mono text-xs leading-loose text-muted">
          {settingsText}
        </div>
      </div>

      <div>
        <div className="mb-2 font-mono text-xs tracking-[.12em] text-muted">{tr.promptLabel}</div>
        <div className="whitespace-pre-wrap rounded-[10px] border border-line bg-panel px-4 py-3.5 text-sm leading-[1.7] text-ink">
          {promptFromSettings(settings)}
        </div>
      </div>

      <div>
        <div className="mb-2 font-mono text-xs tracking-[.12em] text-muted">{tr.lyricsLabel}</div>
        <div className="whitespace-pre-wrap rounded-[10px] border border-line bg-panel px-4 py-3.5 font-mono text-sm leading-[1.8] text-muted">
          {lyricsFromSettings(settings)}
        </div>
      </div>

      <div className="font-mono text-xs text-muted2">
        {tr.aiGenerated} ·{' '}
        <a
          href="https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/main/LICENSE"
          target="_blank"
          rel="noreferrer"
          className="underline underline-offset-2 hover:text-muted"
        >
          MiniMax-Music3 Community License
        </a>
      </div>
    </div>
  );
}
