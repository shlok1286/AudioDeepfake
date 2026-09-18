'use client'

import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  ArrowUp,
  Check,
  ChevronDown,
  CircleAlert,
  Cpu,
  FileAudio,
  History,
  Layers,
  Mail,
  Mic2,
  Play,
  RefreshCw,
  RotateCcw,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  UploadCloud,
  X,
} from 'lucide-react'
import { analyzeAudio, getMockPreviewResult, type AnalysisResult } from '@/lib/analyzeAudio'
import { clearAllHistory, deleteHistoryItem, getHistoryItems, saveHistoryItem, type HistoryItem } from '@/lib/historyDb'
import GradientWaves from '@/components/GradientWaves'
import AudioPlayer from '@/components/AudioPlayer'
import HistoryView from '@/components/HistoryView'
import ResultCard from '@/components/ResultCard'
import AiLoader from '@/components/ui/ai-loader'
import Navbar from '@/components/ui/navbar'

const SUPPORTED_EXTENSIONS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
const MAX_FILE_SIZE = 50 * 1024 * 1024 // 50MB

const PROCESSING_STAGES = [
  { label: 'Analyzing audio...', desc: 'Audio recording loaded & standardized to 16 kHz mono' },
  { label: 'Extracting acoustic features...', desc: 'Computing 128-Mel spectrogram time-frequency windows' },
  { label: 'Running voice verification...', desc: 'Executing 4-layer CNN inference & Grad-CAM attribution' },
]

const faqs = [
  ['What audio formats are supported?', 'VerifyVoice accepts WAV, MP3, FLAC, OGG, and M4A audio files up to 50MB.'],
  ['Is my audio stored?', 'No. Audio files are analyzed in memory and discarded immediately after inference.'],
  ['How is confidence calculated?', 'Confidence represents the model’s classification probability based on spectral and temporal speech features. It serves as a decision support aid, not absolute certainty.'],
  ['Can AI-generated audio fool the detector?', 'Detection models continuously improve, but adversarial noise, extreme compression, or heavy audio editing can affect confidence scores.'],
]

function formatSeconds(seconds: number | null | undefined): string {
  if (!seconds || isNaN(seconds)) return '--:--'
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m < 10 ? '0' : ''}${m}:${s < 10 ? '0' : ''}${s}`
}

export default function Page() {
  const [activeTab, setActiveTab] = useState<'home' | 'history'>('home')
  const [status, setStatus] = useState<'idle' | 'fileSelected' | 'analyzing' | 'result' | 'error'>('idle')
  const [file, setFile] = useState<File | Blob | null>(null)
  const [audioDuration, setAudioDuration] = useState<number | null>(null)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [historyItems, setHistoryItems] = useState<HistoryItem[]>([])
  const [quotaMessage, setQuotaMessage] = useState<string | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [openFaq, setOpenFaq] = useState<number | null>(0)
  const [showScrollTop, setShowScrollTop] = useState(false)
  const [isDragOver, setIsDragOver] = useState(false)
  const [activeStage, setActiveStage] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    const onScroll = () => {
      setShowScrollTop(window.scrollY > 350)
    }
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  // Animate stages during analyzing
  useEffect(() => {
    if (status !== 'analyzing') {
      setActiveStage(0)
      return
    }
    const interval = setInterval(() => {
      setActiveStage((prev) => (prev < PROCESSING_STAGES.length - 1 ? prev + 1 : prev))
    }, 1100)
    return () => clearInterval(interval)
  }, [status])

  const scrollToSection = (id: string) => {
    if (activeTab !== 'home') {
      setActiveTab('home')
      setTimeout(() => {
        const el = document.getElementById(id)
        if (el) {
          el.scrollIntoView({ behavior: 'smooth', block: 'start' })
        } else {
          window.scrollTo({ top: 0, behavior: 'smooth' })
        }
      }, 70)
    } else {
      const el = document.getElementById(id)
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' })
      } else {
        window.scrollTo({ top: 0, behavior: 'smooth' })
      }
    }
  }

  const refreshHistory = async () => {
    const items = await getHistoryItems()
    setHistoryItems(items)
  }

  useEffect(() => {
    refreshHistory()
  }, [])

  const chooseFile = (next: File | undefined) => {
    setErrorMessage(null)
    if (!next) return

    const nameLower = next.name.toLowerCase()
    const isSupportedExt = SUPPORTED_EXTENSIONS.some((ext) => nameLower.endsWith(ext))
    const isAudioMime = next.type.startsWith('audio/')

    if (!isSupportedExt && !isAudioMime) {
      setErrorMessage(
        `Unsupported audio format "${next.name}". Supported formats: WAV, MP3, FLAC, OGG, M4A.`
      )
      return
    }

    if (next.size > MAX_FILE_SIZE) {
      setErrorMessage(
        `File is too large (${(next.size / (1024 * 1024)).toFixed(1)}MB). Maximum allowed size is 50MB.`
      )
      return
    }

    setFile(next)
    setAudioDuration(null)
    setStatus('fileSelected')

    // Read audio duration from metadata
    try {
      const url = URL.createObjectURL(next)
      const audio = new Audio(url)
      audio.onloadedmetadata = () => {
        if (audio.duration && !isNaN(audio.duration)) {
          setAudioDuration(audio.duration)
        }
        URL.revokeObjectURL(url)
      }
      audio.onerror = () => {
        URL.revokeObjectURL(url)
      }
    } catch {
      // ignore
    }
  }

  const reset = () => {
    setStatus('idle')
    setFile(null)
    setAudioDuration(null)
    setResult(null)
    setQuotaMessage(null)
    setErrorMessage(null)
  }

  const runAnalysis = async () => {
    if (!file || status === 'analyzing') return
    setStatus('analyzing')
    setQuotaMessage(null)
    setErrorMessage(null)

    try {
      const res = await analyzeAudio(file as File, audioDuration || undefined)
      setResult(res)
      setStatus('result')

      // Save to IndexedDB
      const filename = (file as File).name || 'audio_recording.wav'
      const fileSize = file.size || 0
      const saveRes = await saveHistoryItem(file, filename, fileSize, res)

      if (!saveRes.success && saveRes.error) {
        setQuotaMessage(saveRes.error)
      }
      refreshHistory()
    } catch (err: any) {
      console.error('Analysis error:', err)
      setErrorMessage(err.message || 'Analysis could not be completed.')
      setStatus('error')
    }
  }

  const handleSelectHistoryItem = (item: HistoryItem) => {
    setFile(item.audioBlob)
    setAudioDuration(item.duration || null)
    setResult(item.result)
    setStatus('result')
    setActiveTab('home')
    scrollToSection('top')
  }

  const handleDeleteHistoryItem = async (id: string): Promise<boolean> => {
    const success = await deleteHistoryItem(id)
    if (success) {
      await refreshHistory()
      return true
    }
    return false
  }

  const handleClearAllHistory = async (): Promise<boolean> => {
    const success = await clearAllHistory()
    if (success) {
      await refreshHistory()
      return true
    }
    return false
  }

  return (
    <main className="flex min-h-[100svh] w-full flex-col bg-background p-3 sm:p-4 text-foreground">
      <div className="relative min-h-[900px] w-full overflow-clip rounded-[28px] border border-border-glass bg-panel shadow-[0_0_80px_rgba(0,55,180,.12)] sm:min-h-[1000px] sm:rounded-[32px]">
        <div className="ambient ambient-one" />
        <div className="ambient ambient-two" />

        {/* REDESIGNED FLOATING WHITE PILL NAVBAR */}
        <Navbar
          activeTab={activeTab}
          onSelectTab={setActiveTab}
          onScrollToSection={scrollToSection}
        />

        {/* TAB CONTENTS */}
        {activeTab === 'history' ? (
          <div className="relative z-10">
            <HistoryView
              items={historyItems}
              onSelectHistoryItem={handleSelectHistoryItem}
              onDeleteItem={handleDeleteHistoryItem}
              onClearAll={handleClearAllHistory}
              onNavigateToUpload={() => scrollToSection('top')}
            />
          </div>
        ) : (
          <>
            {/* HERO & DETECTION SECTION */}
            <section
              id="top"
              className="relative z-10 flex flex-col items-center overflow-hidden px-4 pb-20 pt-10 text-center sm:px-8 sm:pt-16 scroll-mt-28"
            >
              <div className="pointer-events-none absolute inset-0 z-0">
                <GradientWaves
                  horizonColor="#020308"
                  waveColor="#0b1630"
                  crestColor="#2563eb"
                  speed={0.4}
                  amplitude={2.5}
                  waveScale={0.6}
                  waveRatio={0.9}
                  swell={35}
                  turbulence={20}
                  tilt={1.11}
                  zoom={1.0}
                  height={5.5}
                  fogDepth={15}
                  detail="medium"
                  brightness={1.0}
                  opacity={0.7}
                  mouseInteraction={true}
                  parallaxStrength={0.5}
                  grain={true}
                  grainIntensity={0.05}
                />
              </div>

              <p className="mb-4 text-xs font-medium uppercase tracking-[.24em] text-accent-bright">
                AUDIO AUTHENTICITY, CLARIFIED
              </p>
              <h1 className="max-w-3xl text-balance text-4xl font-semibold tracking-[-.045em] text-gradient sm:text-6xl">
                Detect AI-generated voice — instantly, accurately.
              </h1>
              <p className="mt-5 max-w-xl text-pretty text-sm leading-6 text-secondary sm:text-base">
                Upload a speech recording to determine whether the voice is REAL or FAKE. Evaluated with deep convolutional spectral analysis and Grad-CAM decision attribution.
              </p>

              {/* UPLOAD & ANALYSIS WORKFLOW CONTAINER (SLIGHTLY OFFSET TO THE LEFT) */}
              <div className="mt-10 flex w-full max-w-3xl flex-col items-center justify-center sm:-translate-x-10 md:-translate-x-14 transition-transform">
                {/* Global Error Banner for File Validation */}
                {errorMessage && status !== 'error' && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mb-4 flex w-full max-w-xl items-center justify-between gap-3 rounded-2xl border border-rose-500/40 bg-rose-500/10 p-3.5 text-xs text-rose-300 backdrop-blur-md"
                  >
                    <div className="flex items-center gap-2 text-left">
                      <AlertCircle size={16} className="shrink-0 text-rose-400" />
                      <span>{errorMessage}</span>
                    </div>
                    <button
                      onClick={() => setErrorMessage(null)}
                      className="rounded-full p-1 text-rose-300 transition hover:bg-rose-500/20"
                      aria-label="Dismiss error"
                    >
                      <X size={14} />
                    </button>
                  </motion.div>
                )}

                <AnimatePresence mode="wait">
                  {/* 1. IDLE: DRAG & DROP UPLOAD ZONE */}
                  {status === 'idle' && (
                    <motion.div
                      key="idle"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      className="w-full flex justify-center"
                    >
                      <div
                        onClick={() => inputRef.current?.click()}
                        onDragOver={(e) => {
                          e.preventDefault()
                          setIsDragOver(true)
                        }}
                        onDragLeave={() => setIsDragOver(false)}
                        onDrop={(e) => {
                          e.preventDefault()
                          setIsDragOver(false)
                          chooseFile(e.dataTransfer.files[0])
                        }}
                        className={`upload-zone cursor-pointer transition-all ${
                          isDragOver
                            ? 'border-accent-bright bg-accent/15 scale-[1.01] shadow-[0_0_50px_rgba(23,103,255,0.3)]'
                            : ''
                        }`}
                      >
                        <div className="grid size-14 place-items-center rounded-2xl bg-accent/15 text-accent-bright">
                          <UploadCloud size={32} />
                        </div>
                        <div className="flex flex-col gap-1 text-center">
                          <span className="text-base font-semibold tracking-tight text-foreground">
                            Drag & drop speech audio here or click to browse
                          </span>
                          <span className="text-xs text-secondary">
                            Supported: WAV, MP3, FLAC, OGG, M4A • Maximum 50MB
                          </span>
                        </div>
                      </div>
                    </motion.div>
                  )}

                  {/* 2. FILE SELECTED: AUDIO PREVIEW & ANALYZE BUTTON */}
                  {status === 'fileSelected' && file && (
                    <motion.div
                      key="selected"
                      layout
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      className="w-full max-w-2xl rounded-3xl border border-border-glass bg-surface/50 p-6 sm:p-7 backdrop-blur-xl shadow-2xl text-left"
                    >
                      {/* File Details Header */}
                      <div className="flex items-center justify-between border-b border-border-glass/60 pb-4">
                        <div className="flex items-center gap-3.5 min-w-0">
                          <span className="grid size-11 shrink-0 place-items-center rounded-2xl bg-accent/15 text-accent-bright">
                            <FileAudio size={22} />
                          </span>
                          <div className="min-w-0">
                            <p className="truncate text-sm font-semibold text-foreground">
                              {(file as File).name || 'voice_recording.wav'}
                            </p>
                            <p className="text-xs text-secondary mt-0.5">
                              {(file.size / 1024 / 1024).toFixed(2)} MB • Duration: {formatSeconds(audioDuration)}
                            </p>
                          </div>
                        </div>

                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => inputRef.current?.click()}
                            className="rounded-full border border-border-glass bg-background/50 px-3 py-1.5 text-xs text-secondary transition hover:border-accent hover:text-foreground cursor-pointer"
                          >
                            Change
                          </button>
                          <button
                            onClick={reset}
                            className="rounded-full p-2 text-secondary transition hover:bg-surface hover:text-foreground cursor-pointer"
                            aria-label="Remove file"
                          >
                            <X size={16} />
                          </button>
                        </div>
                      </div>

                      {/* Native / Custom Audio Player for pre-analysis verification */}
                      <div className="mt-4">
                        <span className="text-[11px] font-semibold uppercase tracking-wider text-secondary">
                          Audio Preview (Verify Recording)
                        </span>
                        <AudioPlayer file={file} previewMode={true} />
                      </div>

                      {/* Prominent Analyze Audio Button */}
                      <div className="mt-6 pt-4 border-t border-border-glass/40">
                        <button
                          onClick={runAnalysis}
                          className="primary-button w-full py-3.5 text-sm font-semibold tracking-wide cursor-pointer disabled:opacity-50"
                        >
                          <Play size={16} fill="currentColor" /> Analyze Audio
                        </button>
                        <p className="mt-2 text-center text-[11px] text-secondary">
                          Files are analyzed in memory. Results include REAL/FAKE verdict, confidence score, and spectral attribution.
                        </p>
                      </div>
                    </motion.div>
                  )}

                  {/* 3. PROCESSING STATE: ANIMATION & PIPELINE STAGES */}
                  {status === 'analyzing' && (
                    <motion.div
                      key="analyzing"
                      initial={{ opacity: 0, scale: 0.98 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.98 }}
                      className="w-full max-w-2xl rounded-3xl border border-border-glass bg-panel/80 p-8 shadow-2xl backdrop-blur-xl text-left"
                    >
                      <div className="flex flex-col items-center justify-center text-center">
                        <AiLoader size={90} text="Analyzing" />
                        <h3 className="mt-5 text-lg font-semibold tracking-tight text-foreground">
                          {PROCESSING_STAGES[activeStage]?.label || 'Analyzing audio...'}
                        </h3>
                        <p className="text-xs text-secondary mt-1">
                          {PROCESSING_STAGES[activeStage]?.desc || 'Processing your recording through the deepfake acoustic pipeline...'}
                        </p>
                      </div>

                      {/* Structured Pipeline Stages */}
                      <div className="mt-8 flex flex-col gap-2 rounded-2xl border border-border-glass/60 bg-surface/40 p-4">
                        <span className="text-[10px] font-semibold uppercase tracking-wider text-secondary mb-1">
                          Pipeline Progress
                        </span>
                        {PROCESSING_STAGES.map((stg, i) => {
                          const isDone = i < activeStage
                          const isCurrent = i === activeStage
                          return (
                            <div
                              key={stg.label}
                              className={`flex items-center justify-between rounded-xl px-3 py-2 text-xs transition ${
                                isCurrent
                                  ? 'bg-accent/15 text-accent-bright font-medium'
                                  : isDone
                                  ? 'text-foreground/80'
                                  : 'text-secondary/50'
                              }`}
                            >
                              <div className="flex items-center gap-2.5">
                                <span
                                  className={`grid size-5 place-items-center rounded-full text-[10px] font-mono ${
                                    isDone
                                      ? 'bg-emerald-500/20 text-emerald-400'
                                      : isCurrent
                                      ? 'bg-accent text-white'
                                      : 'bg-surface text-secondary/50'
                                  }`}
                                >
                                  {isDone ? <Check size={12} /> : i + 1}
                                </span>
                                <span>{stg.label}</span>
                              </div>
                              <span className="text-[11px] text-secondary">{stg.desc}</span>
                            </div>
                          )
                        })}
                      </div>

                      <div className="mt-6 flex justify-center">
                        <button
                          onClick={reset}
                          className="rounded-full border border-border-glass bg-surface/50 px-5 py-2 text-xs text-secondary transition hover:bg-surface hover:text-foreground cursor-pointer"
                        >
                          Cancel analysis
                        </button>
                      </div>
                    </motion.div>
                  )}

                  {/* 4. RESULT STATE: STRONG RESULT CARD */}
                  {status === 'result' && result && (
                    <div key="result" className="w-full flex justify-center">
                      <ResultCard
                        result={result}
                        file={file}
                        onReset={reset}
                        onViewHistory={() => {
                          setActiveTab('history')
                          window.scrollTo({ top: 0, behavior: 'smooth' })
                        }}
                      />
                    </div>
                  )}

                  {/* 5. ERROR STATE: USER-FRIENDLY BACKEND & INFERENCE ERROR */}
                  {status === 'error' && (
                    <motion.div
                      key="error"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      className="w-full max-w-2xl rounded-3xl border border-rose-500/40 bg-surface/70 p-6 sm:p-8 backdrop-blur-xl shadow-2xl text-left"
                    >
                      <div className="flex items-start gap-4">
                        <span className="grid size-12 shrink-0 place-items-center rounded-2xl bg-rose-500/15 text-rose-400">
                          <AlertTriangle size={24} />
                        </span>
                        <div className="flex-1">
                          <h3 className="text-base font-semibold text-foreground">
                            Analysis Could Not Be Completed
                          </h3>
                          <p className="mt-2 text-xs leading-5 text-secondary">
                            {errorMessage || 'The server encountered an error during inference.'}
                          </p>

                          <div className="mt-4 rounded-xl border border-border-glass bg-background/60 p-3 text-[11px] leading-5 text-secondary">
                            <strong className="text-foreground/90">Backend Note:</strong> Make sure your Python / ML inference backend is running (e.g. FastAPI / Flask) and set the endpoint via <code className="text-accent-bright">NEXT_PUBLIC_API_URL</code> in your environment.
                          </div>

                          {/* Action Buttons */}
                          <div className="mt-6 flex flex-wrap gap-3">
                            <button
                              onClick={runAnalysis}
                              className="primary-button py-2.5 px-5 text-xs font-semibold cursor-pointer"
                            >
                              <RefreshCw size={14} /> Retry Inference
                            </button>
                            <button
                              onClick={reset}
                              className="rounded-full border border-border-glass bg-surface px-4 py-2.5 text-xs text-secondary transition hover:text-foreground cursor-pointer"
                            >
                              Choose Another File
                            </button>
                          </div>

                          {/* Developer UI Preview Shortcuts (Transparent Testing) */}
                          <div className="mt-6 border-t border-border-glass/40 pt-4">
                            <span className="text-[10px] font-semibold uppercase tracking-wider text-secondary/70">
                              Developer UI States Preview (Test without active backend)
                            </span>
                            <div className="mt-2 flex gap-2.5">
                              <button
                                onClick={() => {
                                  setResult(getMockPreviewResult('real', audioDuration || 14.2))
                                  setStatus('result')
                                }}
                                className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-3 py-1.5 text-xs text-emerald-400 hover:bg-emerald-500/20 cursor-pointer"
                              >
                                View REAL State
                              </button>
                              <button
                                onClick={() => {
                                  setResult(getMockPreviewResult('fake', audioDuration || 14.2))
                                  setStatus('result')
                                }}
                                className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-1.5 text-xs text-rose-400 hover:bg-rose-500/20 cursor-pointer"
                              >
                                View FAKE State
                              </button>
                            </div>
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Hidden File Input */}
              <input
                ref={inputRef}
                type="file"
                accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a"
                className="sr-only"
                onChange={(e) => {
                  chooseFile(e.target.files?.[0])
                  if (inputRef.current) inputRef.current.value = ''
                }}
              />
            </section>

            {/* HOW IT WORKS (REFINED ML ARCHITECTURE PIPELINE) */}
            <section id="how-it-works" className="relative z-10 border-t border-border-glass px-5 py-20 sm:px-12 scroll-mt-28">
              <SectionHeading
                eyebrow="ML Pipeline & Architecture"
                title="How It Works"
                description="Our end-to-end convolutional pipeline analyzes spectral patterns to separate authentic human voices from synthetic clones."
              />
              <div className="mt-12 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {[
                  [UploadCloud, '1. Audio Upload', 'Ingests speech recordings in WAV, MP3, FLAC, OGG, or M4A formats up to 50MB.'],
                  [SlidersHorizontal, '2. Audio Preprocessing', 'Standardizes sample rate, normalizes amplitude, and trims leading/trailing silence.'],
                  [Layers, '3. 2-Second Audio Chunks', 'Partitions the audio into uniform 2-second windows for high-resolution temporal analysis.'],
                  [Activity, '4. Mel-Spectrogram', 'Transforms raw waveforms into 2D time-frequency acoustic power spectrograms.'],
                  [Cpu, '5. CNN Model Inference', 'Deep convolutional network inspects acoustic features, vocoder signatures, and phase artifacts.'],
                  [ShieldCheck, '6. Real / Fake Prediction', 'Outputs a REAL or FAKE verdict with probability confidence and Grad-CAM time-frequency attribution.'],
                ].map(([Icon, title, copy]) => (
                  <div key={title as string} className="glass-card">
                    <span className="icon-chip">
                      <Icon size={18} />
                    </span>
                    <h3 className="mt-6 text-base font-medium">{title as string}</h3>
                    <p className="mt-2 text-sm leading-6 text-secondary">{copy as string}</p>
                  </div>
                ))}
              </div>
            </section>

            {/* METRICS */}
            <section className="relative z-10 border-t border-border-glass px-5 py-20 text-center sm:px-12">
              <SectionHeading
                eyebrow="Built for responsible listening"
                title="Confidence, with context."
                description="No black-box declarations. Every result comes with a confidence score and the signals that shaped it."
              />
              <div className="mx-auto mt-10 flex max-w-2xl flex-wrap justify-center gap-3">
                <Stat value="96.8%" label="model confidence" />
                <Stat value="5" label="audio formats" />
                <Stat value="< 30s" label="average analysis" />
              </div>
            </section>

            {/* FAQ */}
            <section id="faq" className="relative z-10 border-t border-border-glass px-5 py-20 sm:px-32 scroll-mt-28">
              <SectionHeading eyebrow="Questions, answered" title="Know the limitations." />
              <div className="mt-10 flex flex-col gap-2">
                {faqs.map(([q, a], i) => (
                  <div key={q} className="overflow-hidden rounded-2xl border border-border-glass bg-surface/45">
                    <button
                      className="flex w-full items-center justify-between gap-4 px-5 py-4 text-left text-sm font-medium cursor-pointer"
                      onClick={() => setOpenFaq(openFaq === i ? null : i)}
                      aria-expanded={openFaq === i}
                    >
                      {q}
                      <ChevronDown
                        size={16}
                        className={`shrink-0 text-secondary transition ${openFaq === i ? 'rotate-180' : ''}`}
                      />
                    </button>
                    <AnimatePresence initial={false}>
                      {openFaq === i && (
                        <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }}>
                          <p className="px-5 pb-5 text-sm leading-6 text-secondary">{a}</p>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                ))}
              </div>

              {/* Quick return button to hero section */}
              <div className="mt-10 flex justify-center">
                <button
                  onClick={() => scrollToSection('top')}
                  className="inline-flex items-center gap-2 rounded-full border border-border-glass bg-surface/60 px-5 py-2.5 text-xs font-medium text-accent-bright transition hover:border-accent-bright hover:bg-accent/10 cursor-pointer"
                >
                  <ArrowUp size={14} /> Back to Voice Detector (Hero)
                </button>
              </div>
            </section>

            {/* CONTACT */}
            <section id="contact" className="relative z-10 border-t border-border-glass px-5 py-20 text-center sm:py-24 scroll-mt-28">
              <SectionHeading
                eyebrow="Still curious?"
                title="Get in touch."
                description="Questions about detection, partnerships, or responsible AI? We’d love to hear from you."
              />
              <form className="mx-auto mt-8 flex max-w-2xl flex-col gap-2 rounded-2xl border border-border-glass bg-surface/55 p-2 sm:flex-row">
                <input className="field" placeholder="Name" aria-label="Name" />
                <input className="field" type="email" placeholder="Email" aria-label="Email" />
                <button className="primary-button shrink-0 cursor-pointer">
                  <Mail size={15} /> Contact us
                </button>
              </form>
            </section>
          </>
        )}

        {/* FOOTER */}
        <footer className="relative z-10 flex flex-col items-center justify-between gap-4 border-t border-border-glass px-5 py-7 text-xs text-secondary sm:flex-row sm:px-12">
          <span>© 2026 VerifyVoice</span>
          <div className="flex gap-5">
            <button onClick={() => scrollToSection('faq')} className="hover:text-foreground cursor-pointer">
              FAQ
            </button>
            <button onClick={() => scrollToSection('contact')} className="hover:text-foreground cursor-pointer">
              Contact
            </button>
            <button onClick={() => scrollToSection('top')} className="flex items-center gap-1 hover:text-foreground cursor-pointer">
              <ArrowUp size={12} /> Back to top
            </button>
          </div>
        </footer>
      </div>

      {/* FLOATING QUICK-RETURN BUTTON */}
      <AnimatePresence>
        {showScrollTop && (
          <motion.button
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            onClick={() => scrollToSection('top')}
            aria-label="Back to hero section"
            className="fixed bottom-6 right-6 z-40 flex items-center gap-2 rounded-full border border-border-glass bg-panel/90 px-4 py-2.5 text-xs font-medium text-accent-bright shadow-2xl backdrop-blur-xl transition hover:border-accent hover:bg-surface cursor-pointer"
          >
            <ArrowUp size={15} />
            <span className="hidden sm:inline">Back to Hero</span>
          </motion.button>
        )}
      </AnimatePresence>
    </main>
  )
}

function SectionHeading({ eyebrow, title, description }: { eyebrow: string; title: string; description?: string }) {
  return (
    <div className="mx-auto max-w-2xl text-center">
      <p className="text-xs font-medium uppercase tracking-[.22em] text-accent-bright">{eyebrow}</p>
      <h2 className="mt-4 text-3xl font-semibold tracking-[-.04em] text-gradient sm:text-4xl">{title}</h2>
      {description && <p className="mx-auto mt-4 max-w-xl text-sm leading-6 text-secondary">{description}</p>}
    </div>
  )
}

function Stat({ value, label }: { value: string; label: string }) {
  return (
    <div className="rounded-full border border-border-glass bg-surface/60 px-5 py-3">
      <span className="font-mono text-lg text-accent-bright">{value}</span>
      <span className="ml-2 text-xs text-secondary">{label}</span>
    </div>
  )
}
