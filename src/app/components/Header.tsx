import type { WebGpuCapability } from '../../runtime/model/webgpu-device';
import type { ArtifactCacheUiState } from '../artifact-cache-ui';
import { formatBytes } from '../artifact-cache-ui';
import type { Language, Messages } from '../i18n';

type PillTone = 'good' | 'accent' | 'muted' | 'danger';

type Pill = { slot: string; text: string; tone: PillTone; pulse?: boolean; title?: string };

const DOT_TONES: Record<PillTone, string> = {
  good: 'bg-good',
  accent: 'bg-accent',
  muted: 'bg-muted2',
  danger: 'bg-danger',
};

function modelPill(tr: Messages, cacheState: ArtifactCacheUiState): Pill {
  if (cacheState.operation === 'download') {
    const progress = cacheState.downloadProgress;
    const percent =
      progress && progress.totalBytes > 0
        ? ` · ${Math.round((progress.completedBytes / progress.totalBytes) * 100)}%`
        : '';
    return { slot: 'model', text: `${tr.pDl}${percent}`, tone: 'accent', pulse: true };
  }
  const status = cacheState.status;
  if (!status) return { slot: 'model', text: tr.pChecking, tone: 'muted', pulse: true };
  if (status.state === 'ready')
    return {
      slot: 'model',
      text: `${tr.pCached} · ${formatBytes(status.totalArtifactBytes)}`,
      tone: 'good',
    };
  if (status.state === 'partial') return { slot: 'model', text: tr.pPartial, tone: 'accent' };
  return { slot: 'model', text: tr.pMissing, tone: 'muted' };
}

function pills(
  tr: Messages,
  capability: WebGpuCapability | null,
  cacheState: ArtifactCacheUiState,
  generating: boolean,
): Pill[] {
  const webgpu: Pill =
    capability === null
      ? { slot: 'webgpu', text: tr.pWebgpuChecking, tone: 'muted', pulse: true }
      : capability.supported
        ? { slot: 'webgpu', text: tr.pWebgpu, tone: 'good' }
        : { slot: 'webgpu', text: tr.pWebgpuMissing, tone: 'danger', title: capability.reason };
  const modelReady = cacheState.status?.state === 'ready';
  const state: Pill = generating
    ? { slot: 'state', text: tr.pGen, tone: 'accent', pulse: true }
    : modelReady && capability?.supported
      ? { slot: 'state', text: tr.pReady, tone: 'accent' }
      : { slot: 'state', text: tr.pWait, tone: 'muted' };
  return [webgpu, modelPill(tr, cacheState), state];
}

export type HeaderProps = {
  tr: Messages;
  lang: Language;
  theme: 'dark' | 'light';
  isMobile: boolean;
  isMid: boolean;
  capability: WebGpuCapability | null;
  cacheState: ArtifactCacheUiState;
  generating: boolean;
  onToggleLang: () => void;
  onToggleTheme: () => void;
};

export function Header({
  tr,
  lang,
  theme,
  isMobile,
  isMid,
  capability,
  cacheState,
  generating,
  onToggleLang,
  onToggleTheme,
}: HeaderProps) {
  const allPills = pills(tr, capability, cacheState, generating);
  const shown = isMobile ? allPills.slice(-1) : allPills;
  return (
    <header className="flex h-[52px] flex-none items-center gap-3.5 border-b border-line bg-panel px-4">
      <div className="size-2.5 flex-none rotate-45 rounded-[2px] bg-accent" />
      <div className="flex min-w-0 items-baseline gap-2.5">
        <div className="whitespace-nowrap text-base font-bold tracking-tight">MiniMax Music 3</div>
        <div className="whitespace-nowrap font-mono text-xs tracking-[.1em] text-muted">WEBGPU</div>
        {!isMid && <div className="whitespace-nowrap text-xs text-muted2">{tr.tagline}</div>}
      </div>
      <div className="flex-1" />
      <div className="flex items-center gap-1.5">
        {shown.map((pill) => (
          <div
            key={pill.slot}
            title={pill.title}
            className={`flex items-center gap-1.5 rounded-full border border-line bg-panel2 px-2.5 py-1 ${
              pill.title ? 'cursor-help' : ''
            }`}
          >
            <div
              className={`size-1.5 rounded-full ${DOT_TONES[pill.tone]} ${
                pill.pulse ? 'animate-[pulse_1.4s_ease-in-out_infinite]' : ''
              }`}
            />
            <div className="whitespace-nowrap font-mono text-xs text-muted">{pill.text}</div>
          </div>
        ))}
        <button
          type="button"
          onClick={onToggleLang}
          className="ml-1.5 rounded-full border border-line px-3 py-[5px] font-mono text-xs font-semibold text-muted hover:border-muted2 hover:bg-panel2 hover:text-ink"
        >
          {lang === 'ko' ? 'EN' : 'KO'}
        </button>
        <button
          type="button"
          onClick={onToggleTheme}
          className="rounded-full border border-line px-3 py-[5px] text-xs font-semibold text-muted hover:border-muted2 hover:bg-panel2 hover:text-ink"
        >
          {theme === 'dark' ? 'Light' : 'Dark'}
        </button>
      </div>
    </header>
  );
}

export type MobileView = 'create' | 'studio' | 'library';

export function MobileTabs({
  tr,
  view,
  onChange,
}: {
  tr: Messages;
  view: MobileView;
  onChange: (view: MobileView) => void;
}) {
  const tabs: [MobileView, string][] = [
    ['create', tr.mCreate],
    ['studio', tr.mStudio],
    ['library', tr.mLibrary],
  ];
  return (
    <div className="flex gap-1 border-b border-line bg-panel px-3 py-2">
      {tabs.map(([key, label]) => (
        <button
          key={key}
          type="button"
          onClick={() => onChange(key)}
          className={`flex-1 rounded-lg border border-line py-[7px] text-sm font-semibold ${
            view === key ? 'bg-panel2 text-ink' : 'bg-transparent text-muted hover:bg-panel2/60 hover:text-ink'
          }`}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
