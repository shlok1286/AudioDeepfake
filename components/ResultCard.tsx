'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  History,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  X,
} from 'lucide-react'
import { type AnalysisResult } from '@/lib/analyzeAudio'
import AudioPlayer from '@/components/AudioPlayer'
import PredictionExplanation from '@/components/PredictionExplanation'

interface ResultCardProps {
  result: AnalysisResult
  file: File | Blob | null
  onReset: () => void
  onViewHistory?: () => void
}

export default function ResultCard({
  result,
  file,
  onReset,
  onViewHistory,
}: ResultCardProps) {
  const isFake = result.prediction === 'fake'
  const confidencePercent = Math.round(result.confidence)
  const realProb = result.real_probability !== undefined ? Math.round(result.real_probability) : (isFake ? 100 - confidencePercent : confidencePercent)
  const fakeProb = result.fake_probability !== undefined ? Math.round(result.fake_probability) : (isFake ? confidencePercent : 100 - confidencePercent)
  const chunksCount = result.chunks_analyzed ?? 1

  const [showExplanation, setShowExplanation] = useState<boolean>(true)

  return (
    <motion.div
      initial={{ opacity: 0, y: 15, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -15, scale: 0.98 }}
      transition={{ duration: 0.35, ease: 'easeOut' }}
      className="relative w-full max-w-3xl overflow-hidden rounded-3xl border border-border-glass bg-panel/90 p-6 sm:p-8 shadow-2xl backdrop-blur-xl text-left"
    >
      {/* Close/Reset top button */}
      <button
        onClick={onReset}
        className="absolute right-5 top-5 rounded-full p-2 text-secondary transition hover:bg-surface hover:text-foreground cursor-pointer"
        aria-label="Close result and analyze another file"
      >
        <X size={18} />
      </button>

      {/* HEADER VERDICT BADGE */}
      <div className="flex flex-col gap-4">
        <div className="flex flex-wrap items-center gap-3">
          <div
            className={`inline-flex items-center gap-2 rounded-full px-4 py-1.5 text-xs font-bold uppercase tracking-wider ${
              isFake
                ? 'bg-rose-500/15 text-rose-400 border border-rose-500/30'
                : 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
            }`}
          >
            {isFake ? <AlertTriangle size={15} /> : <ShieldCheck size={15} />}
            <span>{isFake ? 'FAKE AUDIO' : 'REAL AUDIO'}</span>
          </div>

          <span className="text-xs text-secondary">
            {isFake ? 'Synthetic / AI-Generated Voice Detected' : 'Authentic Organic Human Voice'}
          </span>
        </div>

        {/* PRIMARY METRICS GRID */}
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-5 mt-2">
          {/* Prediction */}
          <div className="rounded-2xl border border-border-glass bg-surface/40 p-3.5">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-secondary">
              Prediction
            </span>
            <p className={`mt-1 font-mono text-xl font-bold ${isFake ? 'text-rose-400' : 'text-emerald-400'}`}>
              {isFake ? 'FAKE' : 'REAL'}
            </p>
          </div>

          {/* Confidence */}
          <div className="rounded-2xl border border-border-glass bg-surface/40 p-3.5">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-secondary">
              Confidence
            </span>
            <p className="mt-1 font-mono text-xl font-bold text-foreground">
              {confidencePercent}%
            </p>
          </div>

          {/* Real Probability */}
          <div className="rounded-2xl border border-border-glass bg-surface/40 p-3.5">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-secondary">
              Real probability
            </span>
            <p className="mt-1 font-mono text-xl font-bold text-emerald-400/90">
              {realProb}%
            </p>
          </div>

          {/* Fake Probability */}
          <div className="rounded-2xl border border-border-glass bg-surface/40 p-3.5">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-secondary">
              Fake probability
            </span>
            <p className="mt-1 font-mono text-xl font-bold text-rose-400/90">
              {fakeProb}%
            </p>
          </div>

          {/* Chunks Analyzed */}
          <div className="col-span-2 sm:col-span-1 rounded-2xl border border-border-glass bg-surface/40 p-3.5">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-secondary">
              Chunks analyzed
            </span>
            <p className="mt-1 font-mono text-xl font-bold text-foreground">
              {chunksCount}
            </p>
          </div>
        </div>

        {/* Confidence Track */}
        <div className="mt-1 flex flex-col gap-1.5">
          <div className="h-2 w-full overflow-hidden rounded-full bg-surface">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${Math.min(100, Math.max(0, confidencePercent))}%` }}
              transition={{ duration: 0.8, ease: 'easeOut' }}
              className={`h-full rounded-full ${
                isFake
                  ? 'bg-gradient-to-r from-rose-600 to-rose-400 shadow-[0_0_12px_rgba(244,63,94,0.5)]'
                  : 'bg-gradient-to-r from-emerald-600 to-emerald-400 shadow-[0_0_12px_rgba(16,185,129,0.5)]'
              }`}
            />
          </div>
          <p className="text-[11px] text-secondary">
            Aggregate chunk confidence score based on 16 kHz Mel-spectrogram convolutional acoustic analysis.
          </p>
        </div>
      </div>

      {/* AUDIO PLAYER & WAVEFORM */}
      {file && (
        <div className="mt-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider text-secondary">
              Audio Playback
            </span>
          </div>
          <AudioPlayer file={file} suspiciousRegions={result.suspiciousRegions} />
        </div>
      )}

      {/* VIEW EXPLANATION TOGGLE BUTTON */}
      <div className="mt-6">
        <button
          onClick={() => setShowExplanation(!showExplanation)}
          className="flex w-full items-center justify-between rounded-xl border border-border-glass bg-surface/60 px-4 py-3 text-xs font-semibold text-foreground transition hover:bg-surface cursor-pointer"
        >
          <span className="flex items-center gap-2">
            <Sparkles size={14} className="text-accent-bright" />
            <span>View Explanation</span>
          </span>
          <span className="flex items-center gap-1 text-secondary text-[11px]">
            {showExplanation ? 'Hide' : 'Show'} Grad-CAM Spectrogram
            {showExplanation ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </span>
        </button>

        <AnimatePresence>
          {showExplanation && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.25 }}
              className="overflow-hidden mt-3"
            >
              <PredictionExplanation prediction={result.prediction} explanation={result.explanation} />
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* ACTION BUTTONS */}
      <div className="mt-8 flex flex-col gap-3 sm:flex-row">
        <button onClick={onReset} className="primary-button flex-1 py-3 text-sm cursor-pointer">
          <RotateCcw size={15} /> Analyze another audio
        </button>
        {onViewHistory && (
          <button
            onClick={onViewHistory}
            className="flex items-center justify-center gap-2 rounded-full border border-border-glass bg-surface/60 px-5 py-3 text-sm font-semibold text-secondary transition hover:bg-surface hover:text-foreground cursor-pointer"
          >
            <History size={15} /> View in History
          </button>
        )}
      </div>
    </motion.div>
  )
}
