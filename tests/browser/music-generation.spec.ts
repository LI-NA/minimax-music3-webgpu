import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { chromium, expect, test } from '@playwright/test';

test.setTimeout(4 * 60 * 60_000);

test('generates and decodes the fixed five-second WAV and reuses the combined release', async () => {
  const context = await chromium.launchPersistentContext(
    path.resolve(process.env.MINIMAX_MUSIC_CHROME_PROFILE ?? 'artifacts/music-browser-profile'),
    { channel: 'chrome', headless: false },
  );
  const wasmRequests: string[] = [];
  context.on('request', (request) => {
    if (request.url().includes('ort-wasm-simd-threaded.jspi.wasm')) wasmRequests.push(request.url());
  });
  const page = context.pages()[0] ?? (await context.newPage());
  try {
    await page.goto('http://127.0.0.1:5173/diagnostics');
    const generate = page.getByRole('button', { name: 'Generate five-second music' });
    await generate.click();
    const progress = page.getByTestId('music-progress');
    await expect(page.locator('output')).toContainText(/Autoregressive frames \d+\/125/, {
      timeout: 4 * 60 * 60_000,
    });
    await expect(progress).toHaveAttribute('max', '125');
    await expect(progress).toHaveAttribute('value', /\d+/);
    await page.waitForFunction(
      () =>
        document.querySelector('[data-testid="music-generation-result"]') ||
        document.querySelector('output')?.textContent?.startsWith('Error:'),
      undefined,
      { timeout: 4 * 60 * 60_000 },
    );
    await expect(page.locator('output')).not.toHaveText(/^Error:/);
    const result = page.getByTestId('music-generation-result');
    await expect(result).toContainText('"hiddenBytes": 8192000');
    await expect(result).toContainText('"conditionBytes": 1761280');
    await expect(result).toContainText('"latentBytes": 110080');
    await expect(result).toContainText('"wavBytes": 880684');
    expect(wasmRequests.length).toBeGreaterThan(0);
    expect(wasmRequests.every((url) => url.endsWith('.jspi.wasm?v=0569a267'))).toBe(true);

    const decoded = await page.getByTestId('generated-audio').evaluate(async (audio) => {
      const wav = await fetch((audio as HTMLAudioElement).src).then((response) => response.arrayBuffer());
      const audioContext = new AudioContext({ sampleRate: 44_100 });
      try {
        const buffer = await audioContext.decodeAudioData(wav);
        return {
          sampleRate: buffer.sampleRate,
          channels: buffer.numberOfChannels,
          samples: buffer.length,
        };
      } finally {
        await audioContext.close();
      }
    });
    expect(decoded).toEqual({ sampleRate: 44_100, channels: 2, samples: 220_160 });

    const generatedDirectory = path.resolve('artifacts/generated');
    mkdirSync(generatedDirectory, { recursive: true });
    const metrics = await result.locator('pre').textContent();
    if (!metrics) throw new Error('music generation metrics are missing');
    writeFileSync(path.join(generatedDirectory, 'music-generation-metrics.json'), `${metrics}\n`);
    const downloadPromise = page.waitForEvent('download');
    await page.getByTestId('download-music').click();
    const download = await downloadPromise;
    const wavPath = path.join(generatedDirectory, 'minimax-music3-5s.wav');
    await download.saveAs(wavPath);
    const wav = readFileSync(wavPath);
    expect(wav.byteLength).toBe(880_684);
    expect(wav.toString('ascii', 0, 4)).toBe('RIFF');
    expect(wav.readUInt32LE(4)).toBe(880_676);
    expect(wav.toString('ascii', 8, 12)).toBe('WAVE');
    expect(wav.readUInt16LE(20)).toBe(1);
    expect(wav.readUInt16LE(22)).toBe(2);
    expect(wav.readUInt32LE(24)).toBe(44_100);
    expect(wav.readUInt16LE(34)).toBe(16);
    expect(wav.readUInt32LE(40)).toBe(880_640);

    let longestConstantFrameRun = 1;
    let currentConstantFrameRun = 1;
    let channelsDiffer = false;
    let lateWindowDelta = 0;
    const frames = wav.readUInt32LE(40) / 4;
    const lateWindowStart = frames - 44_100;
    let previousLeft = wav.readInt16LE(44);
    let previousRight = wav.readInt16LE(46);
    for (let frame = 1; frame < frames; frame++) {
      const offset = 44 + frame * 4;
      const left = wav.readInt16LE(offset);
      const right = wav.readInt16LE(offset + 2);
      channelsDiffer ||= left !== right;
      if (left === previousLeft && right === previousRight) currentConstantFrameRun += 1;
      else currentConstantFrameRun = 1;
      longestConstantFrameRun = Math.max(longestConstantFrameRun, currentConstantFrameRun);
      if (frame >= lateWindowStart) lateWindowDelta += Math.abs(left - previousLeft) + Math.abs(right - previousRight);
      previousLeft = left;
      previousRight = right;
    }
    expect(longestConstantFrameRun).toBeLessThan(44_100);
    expect(lateWindowDelta).toBeGreaterThan(0);
    expect(channelsDiffer).toBe(true);

    await generate.click();
    await expect(result).toHaveCount(0);
    await expect(result).toContainText('"artifactFetches": 0', { timeout: 4 * 60 * 60_000 });
  } finally {
    await context.close();
  }
});
