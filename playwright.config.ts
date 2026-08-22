import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/browser',
  workers: 1,
  // Releases are served same-origin by the dev server, so one web server replaces the former
  // per-release artifact servers on ports 5174 through 5178.
  webServer: [
    {
      command: 'npm run dev -- --host 127.0.0.1',
      url: 'http://127.0.0.1:5173',
      reuseExistingServer: false,
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
