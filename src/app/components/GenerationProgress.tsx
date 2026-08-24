import { formatSpan } from '../format';
import { generationPercent, type GenerationView } from '../generation-view';
import type { Messages } from '../i18n';
import type { Track } from '../tracks';

type StatCell = { key: string; value: string; accent?: boolean };

function stageStats(tr: Messages, view: GenerationView): StatCell[] {
  const cells: StatCell[] = [];
  if (view.stageIndex === 0 && view.frames) {
    cells.push({ key: tr.statFrames, value: `${view.frames.completed} / ${view.frames.total}` });
    if (view.frames.rate !== undefined)
      cells.push({ key: tr.statThr, value: `${view.frames.rate.toFixed(1)} fr/s`, accent: true });
  } else if (view.stageIndex === 1 && view.acoustic) {
    cells.push({
      key: tr.statChunk,
      value: `${view.acoustic.completed} / ${view.acoustic.total}`,
    });
  } else if (view.stageIndex === 2 && view.flow) {
    // Chunks keep completing underneath the flow steps, and this stage is the only place that
    // count is still on screen once the display stops following the interleaved reports.
    if (view.acoustic) cells.push({ key: tr.statChunk, value: `${view.acoustic.completed} / ${view.acoustic.total}` });
    cells.push({ key: tr.statFlow, value: `${view.flow.completed} / ${view.flow.total}` });
    if (view.flow.stepMs !== undefined)
      cells.push({ key: tr.statStep, value: `${view.flow.stepMs.toFixed(0)} ms`, accent: true });
  } else if (view.stageIndex === 3 && view.vocoder) {
    cells.push({
      key: tr.statVoc,
      value: `${view.vocoder.completed} / ${view.vocoder.total}`,
      accent: true,
    });
  } else if (view.stageIndex >= 4) {
    cells.push({ key: 'WAV', value: '16-bit PCM', accent: true });
  }
  return cells;
}

export type GenerationProgressProps = {
  tr: Messages;
  track: Track;
  view: GenerationView;
  elapsedMs: number;
  onCancel: () => void;
};

export function GenerationProgress({ tr, track, view, elapsedMs, onCancel }: GenerationProgressProps) {
  const percent = generationPercent(view);
  const elapsed = formatSpan(elapsedMs / 1000);
  const remaining = view.etaMs !== undefined ? formatSpan(view.etaMs / 1000) : '--';
  const stats: StatCell[] = [
    { key: tr.statElapsed, value: elapsed },
    ...stageStats(tr, view),
    { key: tr.statRemaining, value: remaining },
  ];
  const flow = view.flow;
  const flowCellCount = flow ? Math.min(60, flow.total) : 0;
  const flowFraction = flow && flow.total > 0 ? flow.completed / flow.total : 0;

  return (
    <div className="mx-auto flex max-w-[600px] flex-col gap-4">
      <div className="flex items-start gap-3">
        <div className="min-w-0 flex-1">
          <div className="animate-[pulse_1.6s_ease-in-out_infinite] font-mono text-xs tracking-[.12em] text-accent">
            {tr.generating}
          </div>
          <div className="mt-1 text-2xl font-bold tracking-tight">{track.settings.title}</div>
          <div className="mt-1.5 font-mono text-xs text-muted">
            Seed {track.settings.seed} · {track.settings.mode === 'fine' ? tr.fineTab : tr.rawTab} · {tr.target}{' '}
            {track.settings.durationSeconds}s
          </div>
        </div>
        <button
          type="button"
          onClick={onCancel}
          className="flex-none rounded-[9px] border border-line bg-transparent px-4 py-[7px] text-sm font-semibold text-danger hover:border-danger/50 hover:bg-danger/10"
        >
          {tr.cancel}
        </button>
      </div>

      <div>
        <div className="h-1.5 overflow-hidden rounded-[3px] bg-panel2">
          <div
            className="h-full rounded-[3px] bg-accent transition-[width] duration-150"
            style={{ width: `${percent}%` }}
          />
        </div>
        {/* Elapsed and remaining have their own cells below, so the bar only carries the percent. */}
        <div className="mt-2 font-mono text-xs text-accent">{percent}%</div>
      </div>

      <div className="grid grid-cols-[repeat(auto-fit,minmax(110px,1fr))] gap-2">
        {stats.map((cell) => (
          <div key={cell.key} className="rounded-[10px] border border-line bg-panel px-3 py-2.5">
            <div className="font-mono text-xs tracking-[.1em] text-muted2">{cell.key}</div>
            <div
              className={`mt-[5px] whitespace-nowrap font-mono text-base ${cell.accent ? 'text-accent' : 'text-ink'}`}
            >
              {cell.value}
            </div>
          </div>
        ))}
      </div>

      <div className="flex flex-col gap-0.5 rounded-xl border border-line bg-panel p-2">
        {tr.stages.map((name, index) => {
          const active = index === view.stageIndex;
          const finished = index < view.stageIndex;
          return (
            <div key={name} className={`rounded-lg px-3 py-[9px] ${active ? 'bg-accent-soft' : ''}`}>
              <div className="flex items-center gap-3">
                <div
                  className={`w-[18px] font-mono text-xs ${
                    active ? 'text-accent' : finished ? 'text-good' : 'text-muted2'
                  }`}
                >
                  0{index + 1}
                </div>
                <div
                  className={`flex-1 text-base ${
                    active ? 'font-semibold text-ink' : finished ? 'text-ink' : 'text-muted2'
                  }`}
                >
                  {name}
                </div>
                <div className={`font-mono text-xs ${finished ? 'text-good' : 'text-muted'}`}>
                  {finished ? tr.done : active && !view.indeterminate ? `${Math.round(view.stageFraction * 100)}%` : ''}
                </div>
              </div>
              {active && (
                <div className="mb-0.5 ml-[30px] mt-2 h-[3px] overflow-hidden rounded-sm bg-panel2">
                  <div
                    className={`h-full rounded-sm bg-accent transition-[width] duration-150 ${
                      view.indeterminate ? 'animate-[pulse_1.4s_ease-in-out_infinite]' : ''
                    }`}
                    style={{
                      width: view.indeterminate ? '100%' : `${Math.round(view.stageFraction * 100)}%`,
                    }}
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>

      {flow && flowCellCount > 0 && (
        <div className="rounded-xl border border-line bg-panel px-4 py-3.5">
          <div className="mb-2.5 flex items-baseline">
            <div className="font-mono text-xs tracking-[.12em] text-muted2">{tr.flowStepsLabel}</div>
            <div className="flex-1" />
            <div className="font-mono text-xs text-muted">
              {flow.completed} / {flow.total}
            </div>
          </div>
          <div className="flex flex-wrap gap-[3px]">
            {Array.from({ length: flowCellCount }, (_, index) => {
              const fraction = (index + 1) / flowCellCount;
              const tone =
                fraction <= flowFraction
                  ? 'bg-accent'
                  : fraction - flowFraction < 1 / flowCellCount
                    ? 'bg-accent-soft'
                    : 'bg-panel2';
              return <div key={index} className={`size-[11px] rounded-[3px] ${tone}`} />;
            })}
          </div>
        </div>
      )}

      <div className="rounded-xl border border-line bg-input px-4 py-3">
        <div className="mb-2 font-mono text-xs tracking-[.12em] text-muted2">{tr.logLabel}</div>
        {/* Not a live region: the counter rows rewrite themselves several times a second, which a
            screen reader would read out without end. */}
        <div className="whitespace-pre-wrap break-all font-mono text-xs leading-[1.9] text-muted">
          {view.log.map((row) => row.line).join('\n')}
        </div>
      </div>
    </div>
  );
}
