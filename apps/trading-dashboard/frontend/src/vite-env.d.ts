/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_ACTIVE_SYMBOL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
