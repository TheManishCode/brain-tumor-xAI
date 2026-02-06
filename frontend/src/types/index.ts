// API Types
export interface HealthStatus {
  status: 'healthy' | 'unhealthy'
  models_loaded: boolean
  device: string
  ensemble_size: number
  available_models: string[]
  timestamp: string
}

export interface ClassProbability {
  class: string
  probability: number
}

export interface ModelVote {
  model: string
  prediction: string
  confidence: number
  weight: number
}

// Backend API response (actual structure from Flask)
export interface BackendPredictionResponse {
  success: boolean
  predicted_class: string
  metadata: {
    display_name: string
    description: string
    severity: 'normal' | 'moderate' | 'high'
    recommendations: string[]
  }
  confidence: number // Percentage 0-100
  probabilities: Record<string, number> // e.g., { "glioma": 85.5, ... }
  uncertainty: number
  ensemble_info: {
    num_models: number
    tta_augmentations: number
    total_predictions: number
  }
  gradcam?: string // base64 encoded image
  error?: string
}

// Frontend-normalized prediction result
export interface PredictionResult {
  success: boolean
  prediction: {
    class: string
    confidence: number // 0-1 range
    display_name: string
    description: string
    severity: 'normal' | 'moderate' | 'high'
    recommendations: string[]
  }
  probabilities: ClassProbability[]
  model_votes: ModelVote[]
  ensemble_stats: {
    models_used: number
    tta_transforms: number
    agreement_score: number
  }
  gradcam?: {
    heatmap: string  // base64 encoded image
    overlay: string  // base64 encoded image
  }
  processing_time: number
  uncertainty: number
}

// UI Types
export interface TumorClassInfo {
  name: string
  displayName: string
  description: string
  severity: 'normal' | 'moderate' | 'high'
  color: string
  icon: string
  recommendations: string[]
}

export interface UploadedFile {
  file: File
  preview: string
  name: string
  size: number
  type: string
}

export interface AnalysisState {
  status: 'idle' | 'uploading' | 'analyzing' | 'complete' | 'error'
  progress: number
  result: PredictionResult | null
  error: string | null
}

// Accessibility Types
export interface AccessibilitySettings {
  reducedMotion: boolean
  highContrast: boolean
  largeText: boolean
  screenReaderMode: boolean
  focusHighlight: boolean
  fontSize: number      // 100 = 100%, 125 = 125%, etc.
  lineHeight: number    // 1.5, 1.75, 2, etc.
  letterSpacing: number // 0, 0.05, 0.1, etc. (em units)
}

// Model info types
export interface ModelInfo {
  name: string
  architecture: string
  accuracy: number
  f1Score: number
  weight: number
  parameters: string
  description: string
}
