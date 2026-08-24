import path from 'node:path';
import { chromium, expect, test } from '@playwright/test';

test.setTimeout(30 * 60_000);

test('prefills and performs ten GPU-resident cached decodes', async () => {
  const context = await chromium.launchPersistentContext(
    path.resolve(process.env.MINIMAX_CHROME_PROFILE ?? 'artifacts/browser-profile'),
    { channel: 'chrome', headless: false },
  );
  const page = context.pages()[0] ?? (await context.newPage());
  const failedRequests: string[] = [];
  const navigations: string[] = [];
  page.on('framenavigated', (frame) => {
    if (frame === page.mainFrame()) navigations.push(frame.url());
  });
  page.on('requestfailed', (request) => {
    failedRequests.push(
      `${request.url()} range=${request.headers().range ?? 'none'}: ${request.failure()?.errorText ?? 'unknown error'}`,
    );
  });
  try {
    const release = process.env.MINIMAX_GLOBAL_RELEASE ?? 'global';
    await page.goto(`http://127.0.0.1:5173/diagnostics.html?release=${release}`);
    await page.getByRole('button', { name: 'Run Global LLM smoke' }).click();

    const result = page.getByTestId('global-smoke-result');
    const progress = page.locator('output');
    await page.waitForFunction(
      () =>
        document.querySelector('[data-testid="global-smoke-result"]') ||
        document.querySelector('output')?.textContent?.startsWith('Error:'),
      undefined,
      { timeout: 30 * 60_000 },
    );
    const progressText = await progress.textContent();
    if (progressText?.startsWith('Error:'))
      throw new Error(
        `${progressText}\nNavigations:\n${navigations.join('\n')}\nFailed requests:\n${failedRequests.join('\n') || 'none observed'}`,
      );
    await expect(result).toContainText('steps: 10', { timeout: 30 * 60_000 });
    await expect(result).toContainText('finite logits: yes');
    await expect(result).toContainText('KV location: gpu-buffer');
    await page.getByRole('button', { name: 'Run Global LLM smoke' }).click();
    await expect(result).toContainText('"artifactFetches": 0', { timeout: 30 * 60_000 });
  } finally {
    await context.close();
  }
});
