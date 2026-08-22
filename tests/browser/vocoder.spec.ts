import path from 'node:path';
import { chromium, expect, test } from '@playwright/test';

test.setTimeout(30 * 60_000);

test('generates a canonical stereo WAV with the exact vocoder on WebGPU and reuses its release', async () => {
  const context = await chromium.launchPersistentContext(
    path.resolve(process.env.MINIMAX_VOCODER_CHROME_PROFILE ?? 'artifacts/browser-profile-vocoder'),
    { channel: 'chrome', headless: false },
  );
  const page = context.pages()[0] ?? (await context.newPage());
  try {
    await page.goto('http://127.0.0.1:5173/diagnostics/vocoder');
    await page.getByRole('button', { name: 'Run vocoder smoke' }).click({ timeout: 5_000 });

    const result = page.getByTestId('vocoder-smoke-result');
    const progress = page.locator('output');
    await page.waitForFunction(
      () =>
        document.querySelector('[data-testid="vocoder-smoke-result"]') ||
        document.querySelector('output')?.textContent?.startsWith('Error:'),
      undefined,
      { timeout: 30 * 60_000 },
    );
    await expect(progress).not.toHaveText(/^Error:/);
    await expect(result).toContainText('waveform: float32 1, 2, 220160');
    await expect(result).toContainText('finite: yes');
    await expect(result).toContainText('WAV bytes: 880684');
    await expect(result).toContainText('audio: 44100 Hz, 2 channels, 220160 samples, 16-bit PCM');

    await page.getByRole('button', { name: 'Run vocoder smoke' }).click();
    await expect(result).toHaveCount(0);
    await expect(result).toContainText('"artifactFetches": 0', { timeout: 30 * 60_000 });
  } finally {
    await context.close();
  }
});
