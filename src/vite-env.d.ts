/// <reference types="vite/client" />

declare const __MINIMAX_APP_VERSION__: string;
declare const __MINIMAX_APP_REVISION__: string;

interface ImportMetaEnv {
  /** Manifest the built application loads. Unset in development, pinned in `.env.production`. */
  readonly VITE_MINIMAX_MANIFEST_URL?: string;
}
