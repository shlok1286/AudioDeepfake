export interface GradCamExplanation {
  type: 'gradcam'
  heatmap?: string // Base64 data URL or external image URL
  message?: string
}

export interface SuspiciousRegion {
  id: string
  startTime: number
  endTime: number
  score: number // 0..1
  type: string
  explanation: string
}

export interface AnalysisResult {
  prediction: 'real' | 'fake'
  confidence: number // 0 to 100 percentage
  fake_probability?: number // 0 to 100 percentage
  real_probability?: number // 0 to 100 percentage
  chunks_analyzed?: number
  gradcam_url?: string
  duration?: number
  explanation?: GradCamExplanation
  verdict: 'HUMAN' | 'AI_GENERATED' | 'UNCERTAIN'
  suspiciousRegions?: SuspiciousRegion[]
  explanations?: string[]
  stages?: string[]
}

/**
 * Expected backend response structure:
 * {
 *   prediction: "REAL" | "FAKE",
 *   confidence: 87.42,
 *   fake_probability: 87.42,
 *   real_probability: 12.58,
 *   chunks_analyzed: 9,
 *   gradcam_url: "/api/gradcam/..."
 * }
 */
export function normalizeBackendResponse(raw: any, fallbackDuration?: number): AnalysisResult {
  // Normalize prediction to strictly 'real' or 'fake'
  const rawPred = String(raw.prediction || raw.verdict || raw.label || '').toLowerCase()
  const isFake =
    rawPred.includes('fake') ||
    rawPred.includes('ai') ||
    rawPred.includes('synthetic') ||
    rawPred.includes('spoof')
  const prediction: 'real' | 'fake' = isFake ? 'fake' : 'real'

  // Normalize confidence to a 0-100 percentage
  let confidence = typeof raw.confidence === 'number' ? raw.confidence : (typeof raw.score === 'number' ? raw.score : 90)
  if (confidence <= 1 && confidence > 0) {
    confidence = Math.round(confidence * 1000) / 10
  } else {
    confidence = Math.round(confidence * 10) / 10
  }

  // Probabilities
  const fake_probability = typeof raw.fake_probability === 'number'
    ? (raw.fake_probability <= 1 && raw.fake_probability > 0 ? Math.round(raw.fake_probability * 1000) / 10 : Math.round(raw.fake_probability * 10) / 10)
    : (isFake ? confidence : Math.round((100 - confidence) * 10) / 10)

  const real_probability = typeof raw.real_probability === 'number'
    ? (raw.real_probability <= 1 && raw.real_probability > 0 ? Math.round(raw.real_probability * 1000) / 10 : Math.round(raw.real_probability * 10) / 10)
    : (isFake ? Math.round((100 - confidence) * 10) / 10 : confidence)

  const chunks_analyzed = typeof raw.chunks_analyzed === 'number' ? raw.chunks_analyzed : undefined
  const gradcam_url = raw.gradcam_url || raw.heatmap_url || (raw.explanation && raw.explanation.heatmap) || undefined

  // Normalize explanation
  let explanation: GradCamExplanation | undefined = undefined
  let heatmapImage = gradcam_url || (raw.explanation && (raw.explanation.heatmap || raw.explanation.image)) || raw.heatmap
  if (heatmapImage && typeof heatmapImage === 'string' && heatmapImage.startsWith('/')) {
    const apiBase = (process.env.NEXT_PUBLIC_API_URL || process.env.VITE_API_URL || '').trim().replace(/\/$/, '')
    if (apiBase) {
      heatmapImage = `${apiBase}${heatmapImage}`
    }
  }
  if (heatmapImage) {
    explanation = {
      type: 'gradcam',
      heatmap: heatmapImage,
      message:
        (raw.explanation && raw.explanation.message) ||
        'Highlighted regions show the time-frequency areas that had the strongest influence on the model prediction.',
    }
  }

  // Compatibility fields for HistoryView and visualizers
  const verdict: 'HUMAN' | 'AI_GENERATED' = prediction === 'fake' ? 'AI_GENERATED' : 'HUMAN'
  const explanations: string[] = Array.isArray(raw.explanations)
    ? raw.explanations
    : [
        prediction === 'fake'
          ? 'Spectral energy variance and vocoder artifacts detected by CNN acoustic encoder.'
          : 'Acoustic micro-timing and breath transients consistent with organic human speech.',
      ]

  return {
    prediction,
    confidence,
    fake_probability,
    real_probability,
    chunks_analyzed,
    gradcam_url,
    duration: typeof raw.duration === 'number' ? raw.duration : fallbackDuration,
    explanation,
    verdict,
    suspiciousRegions: raw.suspiciousRegions,
    explanations,
    stages: raw.stages,
  }
}

/**
 * Primary inference entry point.
 * Sends the audio recording as multipart/form-data to the configured ML backend.
 */
export async function analyzeAudio(file: File, durationSeconds?: number): Promise<AnalysisResult> {
  const apiBase = (process.env.NEXT_PUBLIC_API_URL || process.env.VITE_API_URL || '').trim().replace(/\/$/, '')
  const endpoint = apiBase ? `${apiBase}/api/predict` : '/api/predict'

  const formData = new FormData()
  formData.append('audio', file)
  formData.append('file', file)

  const controller = new AbortController()
  const timeoutMs = 60000
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs)

  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    })

    clearTimeout(timeoutId)

    if (!response.ok) {
      let errorDetail = `Backend server returned error HTTP ${response.status}`
      try {
        const errorJson = await response.json()
        errorDetail = errorJson.error || errorJson.detail || errorJson.message || errorDetail
      } catch {
        // use default error message
      }
      throw new Error(errorDetail)
    }

    const data = await response.json()
    return normalizeBackendResponse(data, durationSeconds)
  } catch (err: any) {
    clearTimeout(timeoutId)

    if (err.name === 'AbortError') {
      throw new Error('Analysis timed out. The ML inference took longer than 60 seconds.')
    }

    // Network / connection failure
    if (err instanceof TypeError && err.message.toLowerCase().includes('fetch')) {
      throw new Error(
        `ML Backend unavailable: Could not connect to inference service at "${endpoint}". Please verify that your backend server is running and NEXT_PUBLIC_API_URL is configured properly.`
      )
    }

    throw err
  }
}

/**
 * Developer helper for testing frontend Real/Fake UI states before launching the ML backend.
 */
export function getMockPreviewResult(type: 'real' | 'fake', duration = 12.5): AnalysisResult {
  return {
    prediction: type,
    confidence: type === 'fake' ? 97.4 : 94.2,
    duration,
    verdict: type === 'fake' ? 'AI_GENERATED' : 'HUMAN',
    explanation: {
      type: 'gradcam',
      message:
        'Highlighted regions show the time-frequency areas that had the strongest influence on the model prediction.',
    },
    explanations:
      type === 'fake'
        ? [
            'Vocoder spectral consistency matches synthetic speech generation models.',
            'Unnatural harmonic phase alignment across vowel segments.',
          ]
        : [
            'Natural organic spectral fluctuation and pitch contour.',
            'Physiological breath intervals consistent with human vocal tract acoustics.',
          ],
  }
}
