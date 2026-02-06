import { useState, useCallback } from 'react'
import { predictImage, getErrorMessage } from '@/services/api'
import type { AnalysisState } from '@/types'
import { useAccessibility } from '@/context/AccessibilityContext'

interface UseAnalysisOptions {
  useTta?: boolean
  includeGradcam?: boolean
}

export function useAnalysis(options: UseAnalysisOptions = {}) {
  const { useTta = true, includeGradcam = true } = options
  const { announceMessage } = useAccessibility()
  
  const [state, setState] = useState<AnalysisState>({
    status: 'idle',
    progress: 0,
    result: null,
    error: null
  })

  const analyze = useCallback(async (file: File) => {
    setState({
      status: 'uploading',
      progress: 0,
      result: null,
      error: null
    })

    announceMessage('Starting image analysis', 'polite')

    try {
      // Simulate upload progress
      setState(prev => ({ ...prev, progress: 25 }))

      const result = await predictImage(file, {
        useTta,
        includeGradcam,
        onProgress: (progress) => {
          setState(prev => ({ ...prev, progress }))
        }
      })

      setState(prev => ({ ...prev, status: 'analyzing', progress: 75 }))

      // Small delay for UX
      await new Promise(resolve => setTimeout(resolve, 500))

      setState({
        status: 'complete',
        progress: 100,
        result,
        error: null
      })

      announceMessage(
        `Analysis complete. Detected: ${result.prediction.display_name} with ${Math.round(result.prediction.confidence * 100)}% confidence`,
        'assertive'
      )

      return result
    } catch (err) {
      const errorMessage = getErrorMessage(err)
      
      setState({
        status: 'error',
        progress: 0,
        result: null,
        error: errorMessage
      })

      announceMessage(`Analysis failed: ${errorMessage}`, 'assertive')
      
      throw err
    }
  }, [useTta, includeGradcam, announceMessage])

  const reset = useCallback(() => {
    setState({
      status: 'idle',
      progress: 0,
      result: null,
      error: null
    })
  }, [])

  return {
    ...state,
    analyze,
    reset,
    isAnalyzing: state.status === 'uploading' || state.status === 'analyzing'
  }
}
