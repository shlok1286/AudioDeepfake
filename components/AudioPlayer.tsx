'use client'

import { useEffect, useRef, useState } from 'react'
import { AlertCircle, Pause, Play, Volume2, VolumeX } from 'lucide-react'
import { type SuspiciousRegion } from '@/lib/analyzeAudio'
import AudioVisualizer from '@/components/AudioVisualizer'
import SuspiciousRegionsPanel from '@/components/SuspiciousRegionsPanel'

interface AudioPlayerProps {
  file: File | Blob
  suspiciousRegions?: SuspiciousRegion[]
  previewMode?: boolean
}

function formatTime(seconds: number): string {
  if (!seconds || isNaN(seconds)) return '00:00'
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins < 10 ? '0' : ''}${mins}:${secs < 10 ? '0' : ''}${secs}`
}

const SPEED_OPTIONS = [0.75, 1, 1.25, 1.5, 2]

export default function AudioPlayer({ file, suspiciousRegions, previewMode = false }: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)

  const [isPlaying, setIsPlaying] = useState<boolean>(false)
  const [currentTime, setCurrentTime] = useState<number>(0)
  const [duration, setDuration] = useState<number>(0)
  const [volume, setVolume] = useState<number>(1)
  const [isMuted, setIsMuted] = useState<boolean>(false)
  const [playbackRate, setPlaybackRate] = useState<number>(1)
  const [playbackError, setPlaybackError] = useState<boolean>(false)

  // Region selection & region-bounded playback state
  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(
    suspiciousRegions && suspiciousRegions.length > 0 ? suspiciousRegions[0].id : null
  )
  const activeRegionRef = useRef<SuspiciousRegion | null>(null)

  // Manage Object URL lifecycle
  useEffect(() => {
    if (!file) return
    const url = URL.createObjectURL(file)
    setAudioUrl(url)
    setPlaybackError(false)

    return () => {
      URL.revokeObjectURL(url)
    }
  }, [file])

  // Sync volume, muted, and playbackRate changes to HTMLAudioElement
  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return
    audio.volume = isMuted ? 0 : volume
    audio.playbackRate = playbackRate
  }, [volume, isMuted, playbackRate])

  const togglePlay = async () => {
    const audio = audioRef.current
    if (!audio) return

    try {
      if (isPlaying) {
        audio.pause()
        setIsPlaying(false)
        activeRegionRef.current = null
      } else {
        if (audio.ended || audio.currentTime >= audio.duration) {
          audio.currentTime = 0
          setCurrentTime(0)
        }
        await audio.play()
        setIsPlaying(true)
      }
    } catch (err) {
      console.error('Audio playback error:', err)
      setPlaybackError(true)
      setIsPlaying(false)
    }
  }

  const handleSeek = (time: number) => {
    const audio = audioRef.current
    const validTime = Math.max(0, Math.min(time, duration || audio?.duration || 0))
    if (audio) {
      audio.currentTime = validTime
    }
    setCurrentTime(validTime)
    activeRegionRef.current = null
  }

  const handleSelectRegion = (region: SuspiciousRegion) => {
    setSelectedRegionId(region.id)
    handleSeek(region.startTime)
  }

  const handlePlayRegion = async (region: SuspiciousRegion) => {
    setSelectedRegionId(region.id)
    activeRegionRef.current = region

    const audio = audioRef.current
    if (!audio) return

    try {
      audio.currentTime = region.startTime
      setCurrentTime(region.startTime)
      await audio.play()
      setIsPlaying(true)
    } catch (err) {
      console.error('Region playback error:', err)
      setPlaybackError(true)
    }
  }

  const handleTimeUpdate = (e: React.SyntheticEvent<HTMLAudioElement>) => {
    const time = e.currentTarget.currentTime
    setCurrentTime(time)

    // Check region end bounds if playing a specific region
    if (activeRegionRef.current) {
      if (time >= activeRegionRef.current.endTime) {
        e.currentTarget.pause()
        setIsPlaying(false)
        activeRegionRef.current = null
      }
    }
  }

  const handleVolumeChange = (newVol: number) => {
    setVolume(newVol)
    if (newVol > 0 && isMuted) {
      setIsMuted(false)
    }
  }

  const toggleMute = () => {
    setIsMuted((prev) => !prev)
  }

  return (
    <div className="mt-6 flex flex-col gap-4">
      {/* Hidden single HTMLAudioElement master source */}
      {audioUrl && (
        <audio
          ref={audioRef}
          src={audioUrl}
          onLoadedMetadata={(e) => {
            const dur = e.currentTarget.duration
            if (dur && !isNaN(dur)) setDuration(dur)
          }}
          onTimeUpdate={handleTimeUpdate}
          onEnded={() => {
            setIsPlaying(false)
            activeRegionRef.current = null
          }}
          onError={() => {
            setPlaybackError(true)
            setIsPlaying(false)
          }}
        />
      )}

      {/* Audio Error Banner */}
      {playbackError && (
        <div className="flex items-center gap-2 rounded-xl border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-300">
          <AlertCircle size={15} /> Audio playback unavailable
        </div>
      )}

      {/* COMPACT PLAYER TOOLBAR */}
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-border-glass bg-surface/60 p-3.5 backdrop-blur-md">
        {/* Play/Pause & Time Counter */}
        <div className="flex items-center gap-3">
          <button
            onClick={togglePlay}
            disabled={playbackError}
            aria-label={isPlaying ? 'Pause audio' : 'Play audio'}
            className="grid size-9 place-items-center rounded-full bg-accent text-white shadow-md shadow-accent/20 transition hover:scale-105 active:scale-95 disabled:opacity-50"
          >
            {isPlaying ? <Pause size={16} fill="currentColor" /> : <Play size={16} fill="currentColor" className="ml-0.5" />}
          </button>

          <div className="font-mono text-xs tracking-tight text-foreground/90">
            <span>{formatTime(currentTime)}</span>
            <span className="mx-1 text-secondary/60">/</span>
            <span className="text-secondary">{formatTime(duration)}</span>
          </div>
        </div>

        {/* Speed Selector & Volume Control */}
        <div className="flex items-center gap-3">
          {/* Playback Speed dropdown */}
          <div className="flex items-center gap-1 rounded-full border border-border-glass bg-background/60 px-2 py-1">
            <span className="text-[10px] text-secondary select-none pl-1">Speed</span>
            <select
              value={playbackRate}
              onChange={(e) => setPlaybackRate(parseFloat(e.target.value))}
              aria-label="Playback speed"
              className="bg-transparent text-xs font-semibold text-accent-bright outline-none cursor-pointer pr-1"
            >
              {SPEED_OPTIONS.map((rate) => (
                <option key={rate} value={rate} className="bg-surface text-foreground">
                  {rate}x
                </option>
              ))}
            </select>
          </div>

          {/* Volume Mute & Slider */}
          <div className="flex items-center gap-2 rounded-full border border-border-glass bg-background/60 px-3 py-1.5">
            <button
              onClick={toggleMute}
              aria-label={isMuted ? 'Unmute audio' : 'Mute audio'}
              className="text-secondary transition hover:text-foreground"
            >
              {isMuted || volume === 0 ? <VolumeX size={15} /> : <Volume2 size={15} />}
            </button>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={isMuted ? 0 : volume}
              onChange={(e) => handleVolumeChange(parseFloat(e.target.value))}
              aria-label="Volume"
              className="h-1.5 w-16 accent-accent cursor-pointer rounded-lg"
            />
          </div>
        </div>
      </div>

      {/* SYNCHRONIZED VISUALIZERS WITH REGION OVERLAYS */}
      <AudioVisualizer
        file={file}
        currentTime={currentTime}
        duration={duration}
        suspiciousRegions={suspiciousRegions}
        selectedRegionId={selectedRegionId}
        onSeek={handleSeek}
        onSelectRegion={handleSelectRegion}
      />

      {/* SUSPICIOUS REGIONS DETAILS PANEL (Omit in upload preview) */}
      {!previewMode && (
        <SuspiciousRegionsPanel
          suspiciousRegions={suspiciousRegions}
          selectedRegionId={selectedRegionId}
          onSelectRegion={handleSelectRegion}
          onPlayRegion={handlePlayRegion}
        />
      )}
    </div>
  )
}
