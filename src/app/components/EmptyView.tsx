import type { Messages } from '../i18n';

export type EmptyViewProps = {
  tr: Messages;
  onExample: (instrumental: boolean) => void;
};

export function EmptyView({ tr, onExample }: EmptyViewProps) {
  const steps = [
    { number: '01', text: tr.step1, meta: 'Fine-grained / Raw' },
    { number: '02', text: tr.step2, meta: 'section tags' },
    { number: '03', text: tr.step3, meta: 'seed · duration' },
  ];
  return (
    <div className="mx-auto mt-14 flex max-w-[480px] flex-col gap-5">
      <div>
        <div className="font-mono text-xs tracking-[.12em] text-muted">{tr.newTrackLabel}</div>
        <div className="mt-1 text-2xl font-bold">{tr.newLabel}</div>
        <div className="mt-2 text-sm leading-[1.7] text-muted">{tr.newDesc}</div>
      </div>
      <div className="flex flex-col gap-2">
        {steps.map((step) => (
          <div
            key={step.number}
            className="flex items-center gap-3 rounded-[10px] border border-line bg-panel px-3.5 py-3"
          >
            <div className="w-[18px] font-mono text-xs text-accent">{step.number}</div>
            <div className="flex-1 text-sm text-ink">{step.text}</div>
            <div className="font-mono text-xs text-muted2">{step.meta}</div>
          </div>
        ))}
      </div>
      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={() => onExample(false)}
          className="rounded-[9px] border border-line bg-panel px-4 py-[9px] text-sm font-semibold text-ink"
        >
          {tr.exVocal}
        </button>
        <button
          type="button"
          onClick={() => onExample(true)}
          className="rounded-[9px] border border-line bg-panel px-4 py-[9px] text-sm font-semibold text-ink"
        >
          {tr.exInst}
        </button>
      </div>
    </div>
  );
}
