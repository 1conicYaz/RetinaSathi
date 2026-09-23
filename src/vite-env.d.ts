/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_INSFORGE_BASE_URL: string;
  readonly VITE_INSFORGE_ANON_KEY: string;
  readonly VITE_INFERENCE_URL: string;
  readonly VITE_INFERENCE_FUNCTION: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
