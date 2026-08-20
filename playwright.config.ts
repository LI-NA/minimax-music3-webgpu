import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/browser',
  workers: 1,
  webServer: [
    {
      command: 'npm run dev -- --host 127.0.0.1',
      url: 'http://127.0.0.1:5173',
      reuseExistingServer: false,
    },
    {
      command: 'node tools/serve-artifacts.mjs',
      url: 'http://127.0.0.1:5174/manifest.json',
      reuseExistingServer: false,
    },
    {
      command: 'node tools/serve-artifacts.mjs',
      url: 'http://127.0.0.1:5175/manifest.json',
      reuseExistingServer: false,
      env: { ...process.env, MINIMAX_RELEASE: 'rvq', MINIMAX_ARTIFACT_PORT: '5175' },
    },
    {
      command: 'node tools/serve-artifacts.mjs',
      url: 'http://127.0.0.1:5176/manifest.json',
      reuseExistingServer: false,
      env: { ...process.env, MINIMAX_RELEASE: 'condition', MINIMAX_ARTIFACT_PORT: '5176' },
    },
  ],
  use: {
    baseURL: 'http://127.0.0.1:5173',
  },
  projects: [
    {
      name: 'chrome',
      use: { channel: 'chrome', headless: false },
    },
  ],
});
