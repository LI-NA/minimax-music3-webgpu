import { Fragment } from 'react';
import type { ArtifactCacheControls, ArtifactCacheUiState } from '../artifact-cache-ui';
import {
  artifactCacheNoticeMessage,
  artifactErrorMessage,
  formatBytes,
  formatEta,
  formatRate,
  persistenceWarningMessage,
} from '../artifact-cache-ui';
import type { Messages } from '../i18n';

export type DownloadCardProps = {
  tr: Messages;
  cacheState: ArtifactCacheUiState;
  controls: ArtifactCacheControls;
  onDownload: () => void;
  onCancel: () => void;
  onDelete: () => void;
  onRefresh: () => void;
};

const secondaryButton =
  'rounded-lg border border-line bg-transparent px-3.5 py-1.5 text-sm text-muted ' +
  'hover:border-muted2 hover:text-ink';
const dangerButton =
  'rounded-lg border border-line bg-transparent px-3.5 py-1.5 text-sm text-danger ' +
  'hover:border-danger/50 hover:bg-danger/10';

export function DownloadCard({
  tr,
  cacheState,
  controls,
  onDownload,
  onCancel,
  onDelete,
  onRefresh,
}: DownloadCardProps) {
  const status = cacheState.status;
  const downloading = cacheState.operation === 'download';
  const progress = cacheState.downloadProgress;
  const completedBytes = progress?.completedBytes ?? status?.completeArtifactBytes ?? 0;
  const totalBytes = progress?.totalBytes ?? status?.totalArtifactBytes ?? 0;
  const percent = totalBytes > 0 ? Math.floor((completedBytes / totalBytes) * 100) : 0;
  const actionLabel = controls.canRetry ? tr.dlRetry : status?.state === 'partial' ? tr.dlResume : tr.dlAction;
  // Every row stays on screen for the whole download and falls back to a placeholder, so a report
  // that arrives without transfer metrics cannot make a row vanish and reflow the card.
  const details = [
    { label: tr.dlFile, value: progress?.currentFile ?? tr.dlUnknown, title: progress?.currentFile },
    { label: tr.dlSpeed, value: progress?.rate === undefined ? tr.dlUnknown : formatRate(progress.rate) },
    { label: tr.dlRemain, value: progress?.etaMs === undefined ? tr.dlUnknown : formatEta(progress.etaMs) },
  ];
  // The specific reason replaces the generic best-effort note rather than stacking on top of it.
  const persistenceNote = cacheState.persistenceWarning
    ? persistenceWarningMessage(tr, cacheState.persistenceWarning)
    : status?.persistence === 'best-effort'
      ? tr.dlBestEffort
      : null;

  return (
    <div className="mx-auto mt-12 flex max-w-[520px] flex-col gap-[18px] rounded-[14px] border border-line bg-panel px-7 py-[26px]">
      <div>
        <div className="font-mono text-xs tracking-[.12em] text-muted">{tr.firstRun}</div>
        <div className="mt-1 text-xl font-bold">{tr.dlTitle}</div>
      </div>
      {!status && !cacheState.lastError && <div className="text-sm text-muted">{tr.dlChecking}</div>}
      {status && (
        <>
          <div className="flex items-baseline gap-3">
            <div className="font-mono text-3xl font-medium text-accent">{percent}%</div>
            <div className="font-mono text-sm text-muted">
              {formatBytes(completedBytes)} / {formatBytes(totalBytes)}
            </div>
          </div>
          <div className="h-2 overflow-hidden rounded bg-panel2">
            <div
              className="h-full rounded"
              style={{
                width: downloading && !progress ? '100%' : `${percent}%`,
                background: 'repeating-linear-gradient(45deg, var(--accent) 0 10px, var(--accent2) 10px 20px)',
                backgroundSize: '28px 28px',
                animation: downloading ? 'slide 1s linear infinite' : undefined,
              }}
            />
          </div>
          {downloading && (
            <dl className="grid grid-cols-[auto_minmax(0,1fr)] items-baseline gap-x-5 gap-y-2 font-mono text-sm">
              {details.map((detail) => (
                <Fragment key={detail.label}>
                  <dt className="text-muted2">{detail.label}</dt>
                  <dd className="min-w-0 truncate text-muted" title={detail.title}>
                    {detail.value}
                  </dd>
                </Fragment>
              ))}
            </dl>
          )}
        </>
      )}
      {cacheState.lastError && (
        <div role="alert" className="text-sm leading-normal text-danger">
          {artifactErrorMessage(tr, cacheState.lastError.code, tr.errWorker)}
          <div className="mt-1 font-mono text-xs leading-normal text-muted2">{cacheState.lastError.message}</div>
        </div>
      )}
      {status && status.sufficient === false && (
        <div className="text-sm leading-normal text-danger">
          {tr.dlInsufficient}{' '}
          {status.availableBytes !== undefined && `${formatBytes(status.availableBytes)} ${tr.dlAvailable} · `}
          {formatBytes(status.requiredHeadroomBytes)} {tr.dlRequired}
        </div>
      )}
      {persistenceNote && <div className="text-xs leading-normal text-muted2">{persistenceNote}</div>}
      {cacheState.notice && (
        <div aria-live="polite" className="text-xs leading-normal text-muted2">
          {artifactCacheNoticeMessage(tr, cacheState.notice)}
        </div>
      )}
      <div className="text-xs leading-relaxed text-muted2">{tr.dlNote}</div>
      <div className="flex flex-wrap gap-2">
        {(controls.canDownload || controls.canRetry) && (
          <button
            type="button"
            onClick={onDownload}
            className="rounded-lg border-none bg-accent px-4 py-1.5 text-sm font-bold text-accent-fg hover:bg-accent2"
          >
            {actionLabel}
          </button>
        )}
        {controls.canCancel && (
          <button type="button" onClick={onCancel} className={secondaryButton}>
            {tr.dlCancel}
          </button>
        )}
        {controls.canRefresh && (
          <button type="button" onClick={onRefresh} className={secondaryButton}>
            {tr.dlRefresh}
          </button>
        )}
        {controls.canDelete && (
          <button type="button" onClick={onDelete} className={dangerButton}>
            {tr.dlDelete}
          </button>
        )}
      </div>
    </div>
  );
}
