'use client'

import { Activity, Eye, Layers } from 'lucide-react'
import { type GradCamExplanation } from '@/lib/analyzeAudio'

interface PredictionExplanationProps {
  prediction: 'real' | 'fake'
  explanation?: GradCamExplanation
}

export default function PredictionExplanation({
  prediction,
  explanation,
}: PredictionExplanationProps) {
  const hasHeatmap = Boolean(explanation?.heatmap)

  return (
    <div className="flex flex-col gap-4 rounded-2xl border border-border-glass bg-surface/50 p-5 text-left backdrop-blur-md">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="grid size-7 place-items-center rounded-lg bg-accent/20 text-accent-bright">
            <Activity size={16} />
          </span>
          <div>
            <h3 className="text-sm font-semibold text-foreground">Model Explanation</h3>
            <p className="text-xs text-secondary">Acoustic Spectrogram Attribution via Grad-CAM</p>
          </div>
        </div>
        <span className="inline-flex items-center gap-1 rounded-full border border-border-glass bg-surface/70 px-2.5 py-1 text-[11px] font-medium text-secondary">
          <Layers size={12} /> Grad-CAM (512-ch Conv2D)
        </span>
      </div>

      {hasHeatmap ? (
        <div className="flex flex-col gap-3">
          <div className="relative overflow-hidden rounded-xl border border-border-glass bg-background/90 shadow-inner">
            {/* Grad-CAM Heatmap Image */}
            <img
              src={explanation?.heatmap}
              alt="Mel-spectrogram Grad-CAM attribution heatmap"
              className="h-auto max-h-80 w-full object-contain rounded-t-xl"
            />
            {/* Axis hints */}
            <div className="flex justify-between border-t border-border-glass/40 bg-surface/80 px-3 py-1.5 text-[10px] font-mono text-secondary">
              <span>0 Hz (Low freq)</span>
              <span>Time → Frequency Mel-Scale</span>
              <span>8000 Hz (Nyquist freq)</span>
            </div>
          </div>

          <p className="text-xs font-medium leading-5 text-foreground/90">
            Highlighted regions indicate spectrogram areas that contributed to the prediction.
          </p>

          <p className="text-[11px] leading-relaxed text-secondary">
            Features extracted from the final convolutional layer show the acoustic energy and frequency distributions that had the highest influence during inference.
          </p>
        </div>
      ) : (
        <div className="flex flex-col gap-2 rounded-xl border border-dashed border-border-glass/60 bg-surface/30 p-4">
          <p className="text-xs font-medium text-foreground/90">
            Model explanation will appear here once Grad-CAM analysis is loaded.
          </p>
          <p className="text-xs text-secondary">
            Highlighted regions indicate spectrogram areas that contributed to the prediction.
          </p>
        </div>
      )}
    </div>
  )
}
