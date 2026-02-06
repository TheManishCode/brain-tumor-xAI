/**
 * Analysis Context Provider
 * ========================
 * Provides global access to the current analysis results
 * so the chatbot can access them from anywhere in the app.
 */

import { createContext, useContext, useState, useCallback, type ReactNode } from 'react'
import type { PredictionResult } from '@/types'

interface AnalysisContextType {
  currentAnalysis: PredictionResult | null
  setCurrentAnalysis: (result: PredictionResult | null) => void
  clearAnalysis: () => void
  analysisHistory: PredictionResult[]
  addToHistory: (result: PredictionResult) => void
}

const AnalysisContext = createContext<AnalysisContextType | undefined>(undefined)

const MAX_HISTORY = 10

export function AnalysisProvider({ children }: { children: ReactNode }) {
  const [currentAnalysis, setCurrentAnalysisState] = useState<PredictionResult | null>(null)
  const [analysisHistory, setAnalysisHistory] = useState<PredictionResult[]>([])

  const setCurrentAnalysis = useCallback((result: PredictionResult | null) => {
    setCurrentAnalysisState(result)
    if (result) {
      // Also add to history
      setAnalysisHistory(prev => {
        const newHistory = [result, ...prev.filter(r => r !== result)]
        return newHistory.slice(0, MAX_HISTORY)
      })
    }
  }, [])

  const clearAnalysis = useCallback(() => {
    setCurrentAnalysisState(null)
  }, [])

  const addToHistory = useCallback((result: PredictionResult) => {
    setAnalysisHistory(prev => {
      const newHistory = [result, ...prev]
      return newHistory.slice(0, MAX_HISTORY)
    })
  }, [])

  return (
    <AnalysisContext.Provider 
      value={{ 
        currentAnalysis, 
        setCurrentAnalysis, 
        clearAnalysis,
        analysisHistory,
        addToHistory
      }}
    >
      {children}
    </AnalysisContext.Provider>
  )
}

export function useAnalysisContext() {
  const context = useContext(AnalysisContext)
  if (context === undefined) {
    throw new Error('useAnalysisContext must be used within an AnalysisProvider')
  }
  return context
}
