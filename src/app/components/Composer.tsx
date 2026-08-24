import { useRef } from 'react';
import type { MusicSamplingInput } from '../../workers/protocol';
import { DEFAULT_SAMPLING, EXAMPLES, LYRIC_TAGS } from '../examples';
import type { Messages } from '../i18n';
import type { PromptMode } from '../track-store';
import { combineFinePrompt, INSTRUMENTAL_LYRICS } from '../tracks';

export type ComposerState = {
  title: string;
  mode: PromptMode;
  fineMeta: string;
  fineVocal: string;
  fineArrangement: string;
  rawPrompt: string;
  rawTouched: boolean;
  lyrics: string;
  savedLyrics: string;
  instrumental: boolean;
  durationSeconds: number;
  seedInput: string;
  advancedOpen: boolean;
  sampling: MusicSamplingInput;
};

export function createComposerState(): ComposerState {
  return {
    title: '',
    mode: 'fine',
    fineMeta: EXAMPLES.meta,
    fineVocal: EXAMPLES.vocal,
    fineArrangement: EXAMPLES.arrangement,
    rawPrompt: '',
    rawTouched: false,
    lyrics: EXAMPLES.lyrics,
    savedLyrics: '',
    instrumental: false,
    durationSeconds: 30,
    seedInput: '',
    advancedOpen: false,
    sampling: { ...DEFAULT_SAMPLING },
  };
}

const QUICK_DURATIONS = [10, 30, 60, 120, 300];

const inputClass =
  'w-full rounded-lg border border-line bg-input px-2.5 py-2 text-sm text-ink ' +
  'placeholder:text-muted2 enabled:hover:border-muted2/60 focus:border-muted2 focus:outline-none';
const textareaClass =
  'w-full resize-y rounded-lg border border-line bg-input px-2.5 py-2 text-sm leading-relaxed ' +
  'text-ink placeholder:text-muted2 enabled:hover:border-muted2/60 focus:border-muted2 focus:outline-none';
// Heights are a whole number of lines: 22.75px per line plus 18px of padding and border. A field
// that ends mid-line reads as an accident, and the three prompt fields take the same text so they
// take the same room.
const promptFieldClass = `${textareaClass} h-[109px]`;
const numberClass =
  'w-full rounded-[7px] border border-line bg-input px-2 py-[7px] font-mono text-sm ' +
  'text-ink enabled:hover:border-muted2/60 focus:border-muted2 focus:outline-none';

function FieldLabel({ label, tag }: { label: string; tag?: string }) {
  return (
    <div className="mb-1.5 flex items-baseline gap-2">
      <div className="text-sm font-bold">{label}</div>
      {tag && <div className="font-mono text-xs uppercase tracking-[.08em] text-muted2">{tag}</div>}
    </div>
  );
}

function AdvancedSlider({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <div>
      <div className="mb-1 flex items-baseline">
        <div className="font-mono text-xs tracking-[.06em] text-muted">{label}</div>
        <div className="flex-1" />
        <div className="font-mono text-sm">{value}</div>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="w-full"
      />
    </div>
  );
}

function AdvancedNumber({
  label,
  value,
  min,
  max,
  step,
  integer,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max?: number;
  step: number;
  integer?: boolean;
  onChange: (value: number) => void;
}) {
  return (
    <div>
      <div className="mb-1 font-mono text-xs tracking-[.06em] text-muted">{label}</div>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => {
          const parsed = integer ? parseInt(event.target.value, 10) : Number(event.target.value);
          if (Number.isFinite(parsed)) onChange(parsed);
        }}
        className={numberClass}
      />
    </div>
  );
}

export type ComposerProps = {
  tr: Messages;
  composer: ComposerState;
  generating: boolean;
  canGenerate: boolean;
  summary: string;
  notice: string | null;
  onPatch: (patch: Partial<ComposerState>) => void;
  onGenerate: () => void;
};

export function Composer({
  tr,
  composer,
  generating,
  canGenerate,
  summary,
  notice,
  onPatch,
  onGenerate,
}: ComposerProps) {
  const lyricsRef = useRef<HTMLTextAreaElement>(null);
  const seedFixed = composer.seedInput.trim().length > 0;
  const dimmed = composer.instrumental ? 'opacity-55' : '';

  const openRawTab = () => {
    const patch: Partial<ComposerState> = { mode: 'raw' };
    if (!composer.rawTouched && !composer.rawPrompt) patch.rawPrompt = combineFinePrompt(composer);
    onPatch(patch);
  };

  const toggleInstrumental = () => {
    if (composer.instrumental) onPatch({ instrumental: false, lyrics: composer.savedLyrics || EXAMPLES.lyrics });
    else
      onPatch({
        instrumental: true,
        savedLyrics: composer.lyrics,
        lyrics: INSTRUMENTAL_LYRICS,
      });
  };

  const insertTag = (tag: string) => {
    if (composer.instrumental) return;
    const textarea = lyricsRef.current;
    const text = composer.lyrics;
    const at = textarea ? textarea.selectionStart : text.length;
    const insertion = `${at > 0 && text[at - 1] !== '\n' ? '\n' : ''}[${tag}]\n`;
    onPatch({ lyrics: text.slice(0, at) + insertion + text.slice(at) });
    requestAnimationFrame(() => {
      if (!textarea) return;
      textarea.focus();
      textarea.selectionStart = textarea.selectionEnd = at + insertion.length;
    });
  };

  const patchSampling = (patch: Partial<MusicSamplingInput>) =>
    onPatch({ sampling: { ...composer.sampling, ...patch } });

  return (
    <div className="flex flex-col gap-4 p-4 pb-6">
      <div>
        <FieldLabel label={tr.titleLabel} tag={tr.optional} />
        <input
          value={composer.title}
          onChange={(event) => onPatch({ title: event.target.value })}
          placeholder={tr.titlePh}
          className={inputClass}
        />
      </div>

      <div>
        <div className="mb-2 flex items-baseline gap-2">
          <div className="text-sm font-bold">Prompt</div>
          <div className="font-mono text-xs text-muted2">{composer.mode === 'fine' ? tr.activeFine : tr.activeRaw}</div>
        </div>
        <div className="flex gap-[3px] rounded-[9px] border border-line bg-input p-[3px]">
          {(
            [
              ['fine', tr.fineTab, () => onPatch({ mode: 'fine' })],
              ['raw', tr.rawTab, openRawTab],
            ] as const
          ).map(([mode, label, onClick]) => (
            <button
              key={mode}
              type="button"
              onClick={onClick}
              className={`flex-1 rounded-[7px] border-none py-[7px] text-sm font-semibold ${
                composer.mode === mode ? 'bg-panel2 text-ink' : 'bg-transparent text-muted hover:text-ink'
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {composer.mode === 'fine' ? (
          <div className="mt-3 flex flex-col gap-3">
            <div>
              <FieldLabel label={tr.styleLabel} tag={tr.styleTag} />
              <textarea
                value={composer.fineMeta}
                onChange={(event) => onPatch({ fineMeta: event.target.value })}
                placeholder={tr.metaPh}
                className={promptFieldClass}
              />
            </div>
            <div className={dimmed}>
              <div className="mb-1.5 flex items-baseline gap-2">
                <div className="text-sm font-bold">{tr.vocalLabel}</div>
                <div className="font-mono text-xs uppercase tracking-[.08em] text-muted2">{tr.vocalTag}</div>
                {composer.instrumental && (
                  <div className="rounded-full bg-accent-soft px-[7px] py-[2px] font-mono text-xs text-accent">
                    {tr.instBadge}
                  </div>
                )}
              </div>
              <textarea
                value={composer.fineVocal}
                onChange={(event) => onPatch({ fineVocal: event.target.value })}
                disabled={composer.instrumental}
                placeholder={tr.vocalPh}
                className={promptFieldClass}
              />
            </div>
            <div>
              <FieldLabel label={tr.arrLabel} tag={tr.arrTag} />
              <textarea
                value={composer.fineArrangement}
                onChange={(event) => onPatch({ fineArrangement: event.target.value })}
                placeholder={tr.arrPh}
                className={promptFieldClass}
              />
            </div>
            <div className="text-xs leading-normal text-muted2">{tr.fineNote}</div>
          </div>
        ) : (
          <div className="mt-3">
            <textarea
              value={composer.rawPrompt}
              onChange={(event) => onPatch({ rawPrompt: event.target.value, rawTouched: true })}
              placeholder={tr.rawPh}
              className={`${textareaClass} h-[246px]`}
            />
            <div className="mt-1.5 text-xs leading-normal text-muted2">{tr.rawNote}</div>
          </div>
        )}
      </div>

      <div className="h-px bg-line" />

      <div className={dimmed}>
        <div className="mb-2 flex items-center gap-2">
          <div className="text-sm font-bold">{tr.lyricsTitle}</div>
          <div className="flex-1" />
          <button
            type="button"
            onClick={toggleInstrumental}
            className="group flex cursor-pointer items-center gap-2 border-none bg-transparent p-0"
          >
            <div
              className={`relative h-[19px] w-8 rounded-full transition-colors ${
                composer.instrumental ? 'bg-accent group-hover:bg-accent2' : 'bg-line group-hover:bg-muted2/50'
              }`}
            >
              <div
                className="absolute top-[2.5px] size-3.5 rounded-full bg-white shadow transition-[left]"
                style={{ left: composer.instrumental ? 15 : 3 }}
              />
            </div>
            <div className="text-sm text-muted transition-colors group-hover:text-ink">{tr.instrumental}</div>
          </button>
        </div>
        <div className="mb-2 flex flex-wrap gap-1">
          {LYRIC_TAGS.map((tag) => (
            <button
              key={tag}
              type="button"
              onClick={() => insertTag(tag)}
              disabled={composer.instrumental}
              className="rounded-md border border-line bg-panel2 px-2 py-[3px] font-mono text-xs text-muted enabled:hover:border-muted2 enabled:hover:text-ink disabled:opacity-45"
            >
              {tag}
            </button>
          ))}
        </div>
        <textarea
          ref={lyricsRef}
          value={composer.lyrics}
          onChange={(event) => onPatch({ lyrics: event.target.value })}
          disabled={composer.instrumental}
          placeholder={tr.lyricsPh}
          className={`${textareaClass} h-[200px] font-mono`}
        />
        <div className="mt-1.5 text-xs text-muted2">{tr.lyricsNote}</div>
      </div>

      <div className="h-px bg-line" />

      <div>
        <div className="mb-2 flex items-baseline">
          <div className="text-sm font-bold">{tr.maxDur}</div>
          <div className="flex-1" />
          <div className="font-mono text-sm text-accent">{composer.durationSeconds}s</div>
        </div>
        <input
          type="range"
          min={5}
          max={300}
          step={5}
          value={composer.durationSeconds}
          onChange={(event) => onPatch({ durationSeconds: Number(event.target.value) })}
          className="mb-2 w-full"
          aria-label={tr.maxDur}
        />
        <div className="flex gap-[5px]">
          {QUICK_DURATIONS.map((value) => (
            <button
              key={value}
              type="button"
              onClick={() => onPatch({ durationSeconds: value })}
              className={`flex-1 rounded-md border py-1 font-mono text-xs ${
                composer.durationSeconds === value
                  ? 'border-accent bg-accent-soft text-accent'
                  : 'border-line bg-transparent text-muted hover:border-muted2 hover:text-ink'
              }`}
            >
              {value}s
            </button>
          ))}
        </div>
      </div>

      <div>
        <div className="mb-2 flex items-baseline gap-2">
          <div className="text-sm font-bold">{tr.seedTitle}</div>
          <div
            className={`rounded-full px-[7px] py-[2px] font-mono text-xs tracking-[.08em] ${
              seedFixed ? 'bg-accent-soft text-accent' : 'bg-panel2 text-muted'
            }`}
          >
            {seedFixed ? 'FIXED' : 'RANDOM'}
          </div>
        </div>
        <input
          value={composer.seedInput}
          onChange={(event) => onPatch({ seedInput: event.target.value.replace(/[^0-9]/g, '') })}
          inputMode="numeric"
          placeholder={tr.seedPh}
          className={`${inputClass} font-mono`}
        />
        <div className="mt-1.5 text-xs leading-normal text-muted2">{tr.seedNote}</div>
      </div>

      <div>
        <button
          type="button"
          onClick={() => onPatch({ advancedOpen: !composer.advancedOpen })}
          className="group flex w-full items-center gap-2.5 border-none bg-transparent p-0 text-ink"
        >
          <div className="whitespace-nowrap text-sm font-bold">{tr.advanced}</div>
          <div className="h-px flex-1 bg-line transition-colors group-hover:bg-muted2/50" />
          <div
            className={`font-mono text-xs text-muted transition-all duration-150 group-hover:text-accent ${
              composer.advancedOpen ? 'rotate-90' : ''
            }`}
          >
            ▸
          </div>
        </button>
        {composer.advancedOpen && (
          <div className="mt-3.5 flex flex-col gap-3.5">
            <AdvancedSlider
              label="FLOW GUIDANCE"
              value={composer.sampling.flowGuidance}
              min={1}
              max={4}
              step={0.1}
              onChange={(flowGuidance) => patchSampling({ flowGuidance })}
            />
            <AdvancedSlider
              label="FLOW STEPS"
              value={composer.sampling.flowSteps}
              min={4}
              max={60}
              step={1}
              onChange={(flowSteps) => patchSampling({ flowSteps })}
            />
            <div className="grid grid-cols-2 gap-2.5">
              <AdvancedNumber
                label="GLOBAL GUIDANCE"
                value={composer.sampling.globalGuidance}
                min={0}
                step={0.1}
                onChange={(globalGuidance) => patchSampling({ globalGuidance })}
              />
              <AdvancedNumber
                label="TEMPERATURE"
                value={composer.sampling.temperature}
                min={0.05}
                step={0.05}
                onChange={(temperature) => patchSampling({ temperature })}
              />
              <AdvancedNumber
                label="SEMANTIC TOP-K"
                value={composer.sampling.semanticTopK}
                min={1}
                max={16385}
                step={1}
                integer
                onChange={(semanticTopK) => patchSampling({ semanticTopK })}
              />
              <AdvancedNumber
                label="RESIDUAL TOP-K"
                value={composer.sampling.residualTopK}
                min={1}
                max={1024}
                step={1}
                integer
                onChange={(residualTopK) => patchSampling({ residualTopK })}
              />
            </div>
            <button
              type="button"
              onClick={() => onPatch({ sampling: { ...DEFAULT_SAMPLING } })}
              className="self-start rounded-[7px] border border-line bg-transparent px-3 py-[5px] text-sm text-muted hover:border-muted2 hover:text-ink"
            >
              {tr.reset}
            </button>
          </div>
        )}
      </div>

      <div className="mt-0.5">
        <button
          type="button"
          onClick={onGenerate}
          disabled={!generating && !canGenerate}
          className={`w-full rounded-[10px] border-none py-[13px] text-base font-bold ${
            generating
              ? 'bg-transparent text-danger outline outline-1 outline-line hover:bg-danger/10 hover:outline-danger/50'
              : canGenerate
                ? 'bg-accent text-accent-fg hover:bg-accent2 hover:shadow-lg hover:shadow-accent/25'
                : 'bg-panel2 text-muted2'
          }`}
        >
          {generating ? tr.cancelGen : tr.generate}
        </button>
        {notice && (
          <div role="alert" className="mt-2 text-center text-sm text-danger">
            {notice}
          </div>
        )}
        <div className="mt-2 text-center font-mono text-xs text-muted2">{summary}</div>
      </div>
    </div>
  );
}
