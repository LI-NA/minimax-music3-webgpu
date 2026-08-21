type IconProps = { size: number };

export function PlayIcon({ size }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 14 14" aria-hidden="true">
      <path d="M3.5 1.8 L12 7 L3.5 12.2 Z" fill="currentColor" />
    </svg>
  );
}

export function PauseIcon({ size }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 14 14" aria-hidden="true">
      <rect x="2.5" y="2" width="3.2" height="10" rx="1" fill="currentColor" />
      <rect x="8.3" y="2" width="3.2" height="10" rx="1" fill="currentColor" />
    </svg>
  );
}

export function PrevIcon({ size }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 14 14" aria-hidden="true">
      <path d="M11.5 1.8 L4.5 7 L11.5 12.2 Z" fill="currentColor" />
      <rect x="2" y="1.8" width="1.6" height="10.4" fill="currentColor" />
    </svg>
  );
}

export function NextIcon({ size }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 14 14" aria-hidden="true">
      <path d="M2.5 1.8 L9.5 7 L2.5 12.2 Z" fill="currentColor" />
      <rect x="10.4" y="1.8" width="1.6" height="10.4" fill="currentColor" />
    </svg>
  );
}

export function EqBars() {
  return (
    <div className="flex h-[13px] flex-none items-end gap-[2px]" aria-hidden="true">
      <div className="w-[2.5px] rounded-[1px] bg-accent animate-[eq1_.8s_ease-in-out_infinite]" />
      <div className="w-[2.5px] rounded-[1px] bg-accent animate-[eq2_.7s_ease-in-out_infinite]" />
      <div className="w-[2.5px] rounded-[1px] bg-accent animate-[eq3_.9s_ease-in-out_infinite]" />
    </div>
  );
}
