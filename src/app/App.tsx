import { useEffect, useMemo, useReducer, useRef, useState } from 'react';
import type { PointerEvent as ReactPointerEvent, ReactNode } from 'react';
import { inspectWebGpu, type WebGpuCapability } from '../runtime/model/webgpu-device';
import {
  createMusicGenerationRequest,
  type AnyMusicGenerationWorkerResult,
  type MusicGenerationRequest,
  type WorkerResponse,
} from '../workers/protocol';
import { createArtifactCacheClient } from './artifact-cache-client';
import { artifactCacheUiReducer, createArtifactCacheUiState, deriveArtifactCacheControls } from './artifact-cache-ui';
import { Composer, createComposerState, type ComposerState } from './components/Composer';
import { DownloadCard } from './components/DownloadCard';
import { EmptyView } from './components/EmptyView';
import { GenerationProgress } from './components/GenerationProgress';
import { Header, MobileTabs, type MobileView } from './components/Header';
import { Library } from './components/Library';
import { PlayerBar } from './components/PlayerBar';
import { ResultView } from './components/ResultView';
import { EXAMPLES } from './examples';
import { wavFileName } from './format';
import {
  applyGenerationProgress,
  createGenerationView,
  generationPercent,
  type GenerationView,
} from './generation-view';
import { detectLanguage, messages, type Language } from './i18n';
import { resolveManifestUrl } from './manifest-url';
import { deleteStoredTrack, listStoredTracks, saveStoredTrack, type TrackSettings } from './track-store';
import {
  INSTRUMENTAL_LYRICS,
  lyricsFromSettings,
  promptFromSettings,
  randomSeed,
  trackFromStored,
  trackToStored,
  type Track,
} from './tracks';
import { pseudoBars, wavBars } from './waveform';

const MANIFEST_URL = resolveManifestUrl();
const WAVEFORM_BARS = 88;
const SAMPLE_RATE = 44_100;

type Generation = {
  trackId: string;
  startedAt: number;
  view: GenerationView;
};

function createInferenceWorker() {
  return new Worker(new URL('../workers/inference.worker.ts', import.meta.url), { type: 'module' });
}

function readStorage(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writeStorage(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* storage may be unavailable */
  }
}

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}

export function App() {
  const [lang, setLang] = useState<Language>(() => detectLanguage(readStorage('mm3-lang')));
  const [theme, setTheme] = useState<'dark' | 'light'>(() => (readStorage('mm3-theme') === 'light' ? 'light' : 'dark'));
  const tr = messages[lang];

  const [vw, setVw] = useState(() => window.innerWidth);
  const [leftWidth, setLeftWidth] = useState(352);
  const [mobileView, setMobileView] = useState<MobileView>('create');
  const isMobile = vw < 760;
  const isMid = vw < 1180;

  const [capability, setCapability] = useState<WebGpuCapability | null>(null);
  const [cacheState, dispatchCache] = useReducer(artifactCacheUiReducer, null, createArtifactCacheUiState);

  const [composer, setComposer] = useState<ComposerState>(createComposerState);
  const [notice, setNotice] = useState<string | null>(null);
  const [tracks, setTracks] = useState<Track[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [generation, setGeneration] = useState<Generation | null>(null);
  const [playing, setPlaying] = useState(false);
  const [position, setPosition] = useState(0);
  const [volume, setVolume] = useState(80);
  const [loop, setLoop] = useState(false);
  const [bars, setBars] = useState<number[]>([]);

  const mounted = useRef(true);
  const musicWorker = useRef<Worker | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const pendingSeek = useRef<number | null>(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    writeStorage('mm3-theme', theme);
  }, [theme]);

  useEffect(() => {
    document.documentElement.lang = lang;
    writeStorage('mm3-lang', lang);
  }, [lang]);

  useEffect(() => {
    const onResize = () => setVw(window.innerWidth);
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  const cache = useMemo(
    () =>
      createArtifactCacheClient({
        manifestUrl: MANIFEST_URL,
        dispatch: dispatchCache,
        isMounted: () => mounted.current,
        createWorker: createInferenceWorker,
      }),
    [],
  );

  useEffect(() => {
    mounted.current = true;
    void inspectWebGpu(navigator.gpu).then((next) => {
      if (mounted.current) setCapability(next);
    });
    cache.inspect();
    void listStoredTracks()
      .then((stored) => {
        if (!mounted.current) return;
        const restored = stored.map(trackFromStored);
        setTracks(restored);
        setSelectedId((previous) => previous ?? restored[0]?.id ?? null);
      })
      .catch(() => {});
    return () => {
      mounted.current = false;
      cache.terminate();
      musicWorker.current?.terminate();
      musicWorker.current = null;
    };
  }, [cache]);

  const generating = generation !== null;
  const [, tick] = useReducer((count: number) => count + 1, 0);
  useEffect(() => {
    if (!generating) return;
    const interval = setInterval(tick, 500);
    return () => clearInterval(interval);
  }, [generating]);

  const selectedTrack = tracks.find((track) => track.id === selectedId) ?? null;
  const currentTrack = tracks.find((track) => track.id === currentId) ?? null;
  const currentWav = currentTrack?.wav ?? null;
  const readyTracks = tracks.filter((track) => track.status === 'ready');

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    if (!currentWav) {
      audio.pause();
      audio.removeAttribute('src');
      audio.load();
      setPosition(0);
      return;
    }
    const url = URL.createObjectURL(currentWav);
    audio.src = url;
    void audio.play().catch(() => {});
    return () => URL.revokeObjectURL(url);
  }, [currentWav]);

  useEffect(() => {
    if (audioRef.current) audioRef.current.volume = volume / 100;
  }, [volume]);

  useEffect(() => {
    if (audioRef.current) audioRef.current.loop = loop;
  }, [loop]);

  useEffect(() => {
    if (!selectedTrack) {
      setBars([]);
      return;
    }
    if (!selectedTrack.wav) {
      setBars(pseudoBars(selectedTrack.settings.seed, WAVEFORM_BARS));
      return;
    }
    let cancelled = false;
    void wavBars(selectedTrack.wav, WAVEFORM_BARS).then((next) => {
      if (!cancelled) setBars(next);
    });
    return () => {
      cancelled = true;
    };
  }, [selectedTrack]);

  const patchComposer = (patch: Partial<ComposerState>) => {
    setNotice(null);
    setComposer((previous) => ({ ...previous, ...patch }));
  };

  const restoreSettings = (settings: TrackSettings) => {
    setComposer((previous) => ({
      ...previous,
      title: settings.title,
      mode: settings.mode,
      fineMeta: settings.fineMeta,
      fineVocal: settings.fineVocal,
      fineArrangement: settings.fineArrangement,
      rawPrompt: settings.rawPrompt,
      rawTouched: settings.rawPrompt.length > 0,
      lyrics: settings.instrumental ? INSTRUMENTAL_LYRICS : settings.lyrics,
      savedLyrics: settings.instrumental ? settings.lyrics : previous.savedLyrics,
      instrumental: settings.instrumental,
      durationSeconds: settings.durationSeconds,
      seedInput: String(settings.seed),
      sampling: { ...settings.sampling },
    }));
  };

  const settingsFromComposer = (): TrackSettings => {
    const seedText = composer.seedInput.trim();
    return {
      title: composer.title.trim() || `Untitled ${tracks.length + 1}`,
      mode: composer.mode,
      fineMeta: composer.fineMeta,
      fineVocal: composer.fineVocal,
      fineArrangement: composer.fineArrangement,
      rawPrompt: composer.rawPrompt,
      lyrics: composer.instrumental ? composer.savedLyrics : composer.lyrics,
      instrumental: composer.instrumental,
      durationSeconds: composer.durationSeconds,
      seed: seedText ? Math.min(4_294_967_295, Number(seedText)) : randomSeed(),
      sampling: { ...composer.sampling },
    };
  };

  const finishMusicWorker = (active: Worker) => {
    if (musicWorker.current !== active) return;
    musicWorker.current = null;
    active.terminate();
  };

  const failGeneration = (trackId: string, message: string) => {
    setGeneration((previous) => (previous?.trackId === trackId ? null : previous));
    setTracks((previous) =>
      previous.map((track) => (track.id === trackId ? { ...track, status: 'error', error: message } : track)),
    );
  };

  const completeGeneration = (pending: Track, result: AnyMusicGenerationWorkerResult) => {
    const wav = new Blob([result.wav], { type: 'audio/wav' });
    const actualSeconds = result.plan
      ? result.plan.samplesPerChannel / SAMPLE_RATE
      : Math.max(0, result.wav.byteLength - 44) / 4 / SAMPLE_RATE;
    const ready: Track = { ...pending, status: 'ready', actualSeconds, wav };
    setGeneration((previous) => (previous?.trackId === pending.id ? null : previous));
    setTracks((previous) => previous.map((track) => (track.id === pending.id ? ready : track)));
    setCurrentId(pending.id);
    setPosition(0);
    void saveStoredTrack(trackToStored(ready, wav)).catch(() => {});
  };

  const generateFromSettings = (settings: TrackSettings) => {
    const prompt = promptFromSettings(settings);
    if (!prompt) {
      setNotice(tr.promptRequired);
      return;
    }
    const lyrics = lyricsFromSettings(settings);
    if (!lyrics) {
      setNotice(tr.lyricsRequired);
      return;
    }
    let request: MusicGenerationRequest;
    try {
      request = createMusicGenerationRequest({
        manifestUrl: MANIFEST_URL,
        prompt,
        lyrics,
        seed: settings.seed,
        durationSeconds: settings.durationSeconds,
        sampling: settings.sampling,
      });
    } catch (error) {
      setNotice(errorMessage(error, 'Invalid generation request'));
      return;
    }
    musicWorker.current?.terminate();
    musicWorker.current = null;
    let worker: Worker;
    try {
      worker = createInferenceWorker();
    } catch (error) {
      setNotice(errorMessage(error, 'Inference worker failed'));
      return;
    }
    const pending: Track = {
      id: `track-${Date.now()}-${Math.floor(Math.random() * 1_000_000)}`,
      createdAt: Date.now(),
      status: 'generating',
      actualSeconds: 0,
      settings,
    };
    musicWorker.current = worker;
    setNotice(null);
    setTracks((previous) => [pending, ...previous]);
    setSelectedId(pending.id);
    setMobileView('studio');
    setGeneration({ trackId: pending.id, startedAt: Date.now(), view: createGenerationView() });
    worker.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
      if (musicWorker.current !== worker) return;
      if (data.type === 'progress') {
        setGeneration((previous) =>
          previous?.trackId === pending.id
            ? { ...previous, view: applyGenerationProgress(previous.view, data) }
            : previous,
        );
      } else if (data.type === 'music-result') {
        finishMusicWorker(worker);
        completeGeneration(pending, data.result);
      } else if (data.type === 'error') {
        finishMusicWorker(worker);
        failGeneration(pending.id, data.message);
      }
    };
    worker.onerror = (event) => {
      if (musicWorker.current !== worker) return;
      event.preventDefault();
      finishMusicWorker(worker);
      failGeneration(pending.id, errorMessage(event.error ?? new Error(event.message), 'Inference worker failed'));
    };
    try {
      worker.postMessage(request);
    } catch (error) {
      finishMusicWorker(worker);
      failGeneration(pending.id, errorMessage(error, 'Inference worker failed'));
    }
  };

  const cancelGeneration = () => {
    if (!generation) return;
    musicWorker.current?.terminate();
    musicWorker.current = null;
    const trackId = generation.trackId;
    setGeneration(null);
    setTracks((previous) => previous.map((track) => (track.id === trackId ? { ...track, status: 'canceled' } : track)));
  };

  const modelReady = cacheState.status?.state === 'ready';
  const canGenerate = capability?.supported === true && modelReady && !generating;

  const startGeneration = () => {
    if (generating) {
      cancelGeneration();
      return;
    }
    if (!canGenerate) return;
    generateFromSettings(settingsFromComposer());
  };

  const generateVariation = (source: Track) => {
    if (!canGenerate) return;
    restoreSettings(source.settings);
    patchComposer({ seedInput: '' });
    generateFromSettings({
      ...source.settings,
      seed: randomSeed(),
      sampling: { ...source.settings.sampling },
    });
  };

  const selectTrack = (track: Track) => {
    setSelectedId(track.id);
    restoreSettings(track.settings);
    if (isMobile) setMobileView('studio');
  };

  const togglePlayback = () => {
    const audio = audioRef.current;
    if (!audio || !currentTrack) return;
    if (audio.paused) void audio.play().catch(() => {});
    else audio.pause();
  };

  const playToggleTrack = (track: Track) => {
    if (track.status !== 'ready') return;
    if (track.id === currentId) togglePlayback();
    else {
      setCurrentId(track.id);
      setPosition(0);
    }
  };

  const seekToFraction = (fraction: number) => {
    const audio = audioRef.current;
    if (!audio || !currentTrack) return;
    audio.currentTime = fraction * (currentTrack.actualSeconds || audio.duration || 0);
    setPosition(audio.currentTime);
  };

  const waveClick = (fraction: number) => {
    if (!selectedTrack || selectedTrack.status !== 'ready') return;
    if (selectedTrack.id === currentId) {
      seekToFraction(fraction);
      const audio = audioRef.current;
      if (audio?.paused) void audio.play().catch(() => {});
    } else {
      pendingSeek.current = fraction;
      setCurrentId(selectedTrack.id);
      setPosition(0);
    }
  };

  const stepTrack = (delta: number) => {
    if (!readyTracks.length) return;
    const index = readyTracks.findIndex((track) => track.id === currentId);
    const next = readyTracks[(index + delta + readyTracks.length) % readyTracks.length];
    setCurrentId(next.id);
    setSelectedId(next.id);
    setPosition(0);
    restoreSettings(next.settings);
  };

  const deleteTrack = (track: Track) => {
    if (track.status === 'ready' && !window.confirm(tr.delConfirm)) return;
    if (generation?.trackId === track.id) {
      musicWorker.current?.terminate();
      musicWorker.current = null;
      setGeneration(null);
    }
    setTracks((previous) => previous.filter((item) => item.id !== track.id));
    if (selectedId === track.id) setSelectedId(tracks.find((item) => item.id !== track.id)?.id ?? null);
    if (currentId === track.id) {
      setCurrentId(null);
      setPlaying(false);
      setPosition(0);
    }
    void deleteStoredTrack(track.id).catch(() => {});
  };

  const downloadTrack = (track: Track | null) => {
    if (!track?.wav) return;
    const url = URL.createObjectURL(track.wav);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = wavFileName(track.settings.title);
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 5_000);
  };

  const newDraft = () => {
    setSelectedId(null);
    setComposer({
      ...createComposerState(),
      fineMeta: '',
      fineVocal: '',
      fineArrangement: '',
      lyrics: '',
    });
    setNotice(null);
    setMobileView('create');
  };

  const loadExample = (instrumental: boolean) => {
    setSelectedId(null);
    patchComposer(
      instrumental
        ? {
            mode: 'fine',
            fineMeta: EXAMPLES.instrumentalMeta,
            fineVocal: EXAMPLES.instrumentalVocal,
            fineArrangement: EXAMPLES.arrangement,
            lyrics: INSTRUMENTAL_LYRICS,
            savedLyrics: '',
            instrumental: true,
          }
        : {
            mode: 'fine',
            fineMeta: EXAMPLES.meta,
            fineVocal: EXAMPLES.vocal,
            fineArrangement: EXAMPLES.arrangement,
            lyrics: EXAMPLES.lyrics,
            instrumental: false,
          },
    );
    if (isMobile) setMobileView('create');
  };

  const startResize = (event: ReactPointerEvent) => {
    event.preventDefault();
    const startX = event.clientX;
    const startWidth = leftWidth;
    const max = Math.max(280, vw - (isMid ? 260 : 296) - 6 - 380);
    const move = (ev: PointerEvent) => setLeftWidth(Math.min(max, Math.max(280, startWidth + ev.clientX - startX)));
    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  };

  const cacheControls = deriveArtifactCacheControls(cacheState, generating);
  const seedFixed = composer.seedInput.trim().length > 0;
  const summary = [
    `${composer.durationSeconds}s`,
    composer.mode === 'fine' ? tr.fineTab : tr.rawTab,
    composer.instrumental ? tr.inst : tr.vocal,
    seedFixed ? tr.sFixed : tr.sRandom,
    capability !== null && !capability.supported
      ? tr.sWebgpu
      : cacheState.operation === 'download'
        ? tr.sDl
        : modelReady
          ? tr.sReady
          : tr.sModelMissing,
  ].join(' · ');

  const generatingTrack = generation ? (tracks.find((track) => track.id === generation.trackId) ?? null) : null;
  const positionFraction = currentTrack && currentTrack.actualSeconds > 0 ? position / currentTrack.actualSeconds : 0;

  let main: ReactNode;
  if (generation && generatingTrack) {
    main = (
      <GenerationProgress
        tr={tr}
        track={generatingTrack}
        view={generation.view}
        elapsedMs={Date.now() - generation.startedAt}
        onCancel={cancelGeneration}
      />
    );
  } else if (selectedTrack) {
    main = (
      <ResultView
        tr={tr}
        track={selectedTrack}
        bars={bars}
        isCurrent={selectedTrack.id === currentId}
        playing={playing}
        posFraction={selectedTrack.id === currentId ? positionFraction : 0}
        canVariation={canGenerate}
        onWaveClick={waveClick}
        onTogglePlay={() => (selectedTrack.id === currentId ? togglePlayback() : playToggleTrack(selectedTrack))}
        onDownload={() => downloadTrack(selectedTrack)}
        onReuse={() => restoreSettings(selectedTrack.settings)}
        onVariation={() => generateVariation(selectedTrack)}
        onDelete={() => deleteTrack(selectedTrack)}
      />
    );
  } else if (modelReady) {
    main = <EmptyView tr={tr} onExample={loadExample} />;
  } else {
    main = (
      <DownloadCard
        tr={tr}
        cacheState={cacheState}
        controls={cacheControls}
        onDownload={() => void cache.download()}
        onCancel={cache.cancelDownload}
        onDelete={() => {
          if (window.confirm(tr.dlDeleteConfirm)) cache.remove();
        }}
        onRefresh={cache.inspect}
      />
    );
  }

  const gridColumns = isMobile ? 'minmax(0,1fr)' : `${leftWidth}px 6px minmax(0,1fr) ${isMid ? 260 : 296}px`;

  return (
    <div className="flex h-dvh flex-col bg-bg font-sans text-ink">
      <Header
        tr={tr}
        lang={lang}
        theme={theme}
        isMobile={isMobile}
        isMid={isMid}
        capability={capability}
        cacheState={cacheState}
        generating={generating}
        onToggleLang={() => setLang(lang === 'ko' ? 'en' : 'ko')}
        onToggleTheme={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
      />
      {isMobile && <MobileTabs tr={tr} view={mobileView} onChange={setMobileView} />}
      <div className="grid min-h-0 flex-1" style={{ gridTemplateColumns: gridColumns }}>
        <section
          className={`min-h-0 overflow-y-auto border-r border-line bg-panel ${
            !isMobile || mobileView === 'create' ? '' : 'hidden'
          }`}
        >
          <Composer
            tr={tr}
            composer={composer}
            generating={generating}
            canGenerate={canGenerate}
            summary={summary}
            notice={notice}
            onPatch={patchComposer}
            onGenerate={startGeneration}
          />
        </section>
        <div
          onPointerDown={startResize}
          className={`-ml-px cursor-col-resize hover:bg-accent-soft ${isMobile ? 'hidden' : ''}`}
        />
        <main className={`min-h-0 overflow-y-auto p-6 pb-10 ${!isMobile || mobileView === 'studio' ? '' : 'hidden'}`}>
          {main}
        </main>
        <aside
          className={`min-h-0 overflow-y-auto border-l border-line bg-panel ${
            !isMobile || mobileView === 'library' ? '' : 'hidden'
          }`}
        >
          <Library
            tr={tr}
            tracks={tracks}
            selectedId={selectedId}
            currentId={currentId}
            playing={playing}
            generationPercent={generation ? generationPercent(generation.view) : 0}
            onSelect={selectTrack}
            onPlayToggle={playToggleTrack}
            onDelete={deleteTrack}
            onNew={newDraft}
          />
        </aside>
      </div>
      <PlayerBar
        tr={tr}
        track={currentTrack}
        playing={playing}
        position={position}
        volume={volume}
        loop={loop}
        isMobile={isMobile}
        hasReady={readyTracks.length > 0}
        onPrev={() => stepTrack(-1)}
        onNext={() => stepTrack(1)}
        onTogglePlay={togglePlayback}
        onSeek={seekToFraction}
        onVolume={setVolume}
        onToggleLoop={() => setLoop(!loop)}
        onDownload={() => downloadTrack(currentTrack)}
      />
      <audio
        ref={audioRef}
        className="hidden"
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onTimeUpdate={(event) => setPosition(event.currentTarget.currentTime)}
        onLoadedMetadata={(event) => {
          if (pendingSeek.current === null) return;
          event.currentTarget.currentTime = pendingSeek.current * (event.currentTarget.duration || 0);
          pendingSeek.current = null;
        }}
      />
    </div>
  );
}
