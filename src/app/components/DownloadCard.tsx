import type { ArtifactCacheControls, ArtifactCacheUiState } from '../artifact-cache-ui';
import { formatBytes, formatEta, formatRate } from '../artifact-cache-ui';
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

const secondaryButton = 'rounded-lg border border-line bg-transparent px-3.5 py-1.5 text-[11.5px] text-muted';

function detailLine(key: string, value: string) {
  return `${key.padEnd(14)}${value}`;
}

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
  const details = downloading
    ? [
        progress ? detailLine(tr.dlFile, progress.currentFile) : null,
        progress?.rate !== undefined ? detailLine(tr.dlSpeed, formatRate(progress.rate)) : null,
        progress?.etaMs !== undefined ? detailLine(tr.dlRemain, formatEta(progress.etaMs)) : null,
        detailLine(tr.dlVerified, `${formatBytes(completedBytes)} / ${formatBytes(totalBytes)}`),
      ].filter((line): line is string => line !== null)
    : [detailLine(tr.dlVerified, `${formatBytes(completedBytes)} / ${formatBytes(totalBytes)}`)];

  return (
    <div className="mx-auto mt-12 flex max-w-[520px] flex-col gap-[18px] rounded-[14px] border border-line bg-panel px-7 py-[26px]">
      <div>
        <div className="font-mono text-[10px] tracking-[.12em] text-muted">{tr.firstRun}</div>
        <div className="mt-1 text-[19px] font-bold">{tr.dlTitle}</div>
      </div>
      {!status && !cacheState.lastError && <div className="text-xs text-muted">{tr.dlChecking}</div>}
      {status && (
        <>
          <div className="flex items-baseline gap-3">
            <div className="font-mono text-[30px] font-medium text-accent">{percent}%</div>
            <div className="font-mono text-xs text-muted">
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
          <div className="whitespace-pre-wrap font-mono text-[11.5px] leading-loose text-muted">
            {details.join('\n')}
          </div>
        </>
      )}
      {cacheState.lastError && (
        <div role="alert" className="text-[11.5px] leading-normal text-danger">
          {cacheState.lastError.message}
        </div>
      )}
      {status && status.sufficient === false && (
        <div className="text-[11.5px] leading-normal text-danger">
          {tr.dlInsufficient}{' '}
          {status.availableBytes !== undefined && `${formatBytes(status.availableBytes)} ${tr.dlAvailable} · `}
          {formatBytes(status.requiredHeadroomBytes)} {tr.dlRequired}
        </div>
      )}
      {cacheState.persistenceWarning && (
        <div className="text-[11px] leading-normal text-muted2">{cacheState.persistenceWarning}</div>
      )}
      {status?.persistence === 'best-effort' && (
        <div className="text-[11px] leading-normal text-muted2">{tr.dlBestEffort}</div>
      )}
      {cacheState.notice && <div className="text-[11px] leading-normal text-muted2">{cacheState.notice}</div>}
      <div className="text-[11px] leading-relaxed text-muted2">{tr.dlNote}</div>
      <div className="flex flex-wrap gap-2">
        {(controls.canDownload || controls.canRetry) && (
          <button
            type="button"
            onClick={onDownload}
            className="rounded-lg border-none bg-accent px-4 py-1.5 text-[11.5px] font-bold text-accent-fg"
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
          <button type="button" onClick={onDelete} className={`${secondaryButton} text-danger`}>
            {tr.dlDelete}
          </button>
        )}
      </div>
    </div>
  );
}
