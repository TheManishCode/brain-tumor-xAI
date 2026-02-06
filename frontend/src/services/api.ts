import axios from 'axios'
import type { HealthStatus, PredictionResult, BackendPredictionResponse } from '@/types'

const API_BASE = '/api'

// Model weights for display purposes
const MODEL_WEIGHTS: Record<string, number> = {
  'efficientnet_b3': 0.40,
  'resnet50': 0.35,
  'densenet121': 0.25
}

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE,
  timeout: 60000, // 60 seconds for analysis
  headers: {
    'Content-Type': 'application/json',
  },
})

// Health check
export async function checkHealth(): Promise<HealthStatus> {
  const response = await api.get<HealthStatus>('/health')
  return response.data
}

// Transform backend response to frontend format
function transformPredictionResponse(data: BackendPredictionResponse, startTime: number): PredictionResult {
  const processingTime = (Date.now() - startTime) / 1000
  
  // Transform probabilities from object to array
  const probabilities = Object.entries(data.probabilities).map(([className, probability]) => ({
    class: className,
    probability: probability / 100 // Convert from percentage to 0-1
  }))
  
  // Create synthetic model votes based on ensemble info
  const modelVotes = Object.entries(MODEL_WEIGHTS).map(([model, weight]) => ({
    model: model.replace('_', '-'),
    prediction: data.predicted_class,
    confidence: data.confidence / 100,
    weight
  }))
  
  // Calculate agreement score based on uncertainty (inverse relationship)
  const agreementScore = 1 - data.uncertainty
  
  return {
    success: data.success,
    prediction: {
      class: data.predicted_class,
      confidence: data.confidence / 100, // Convert to 0-1 range
      display_name: data.metadata.display_name,
      description: data.metadata.description,
      severity: data.metadata.severity,
      recommendations: data.metadata.recommendations
    },
    probabilities,
    model_votes: modelVotes,
    ensemble_stats: {
      models_used: data.ensemble_info.num_models,
      tta_transforms: data.ensemble_info.tta_augmentations,
      agreement_score: agreementScore
    },
    gradcam: data.gradcam ? {
      heatmap: data.gradcam,
      overlay: data.gradcam
    } : undefined,
    processing_time: processingTime,
    uncertainty: data.uncertainty
  }
}

// Predict tumor classification
export async function predictImage(
  file: File,
  options: {
    useTta?: boolean
    includeGradcam?: boolean
    onProgress?: (progress: number) => void
  } = {}
): Promise<PredictionResult> {
  const startTime = Date.now()
  const formData = new FormData()
  formData.append('file', file)
  
  if (options.useTta !== undefined) {
    formData.append('use_tta', String(options.useTta))
  }
  
  if (options.includeGradcam !== undefined) {
    formData.append('include_gradcam', String(options.includeGradcam))
  }

  const response = await api.post<BackendPredictionResponse>('/predict', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: (progressEvent) => {
      if (progressEvent.total && options.onProgress) {
        const progress = Math.round((progressEvent.loaded * 50) / progressEvent.total)
        options.onProgress(progress)
      }
    },
  })

  return transformPredictionResponse(response.data, startTime)
}

// Advanced analysis with explainability
export interface AdvancedAnalysisResponse {
  success: boolean
  predicted_class: string
  metadata: {
    display_name: string
    description: string
    severity: 'normal' | 'moderate' | 'high'
    recommendations: string[]
  }
  confidence: number
  probabilities: Record<string, number>
  uncertainty: number
  ensemble_info: {
    num_models: number
    tta_augmentations: number
    total_predictions: number
  }
  processing_time: number
  explainability?: {
    predicted_class: string
    confidence: number
    visualizations: {
      gradcam?: { heatmap: string; overlay: string; description: string }
      gradcam_pp?: { heatmap: string; overlay: string; description: string }
      integrated_gradients?: { heatmap: string; overlay: string; description: string }
      saliency?: { heatmap: string; overlay: string; description: string }
      lime?: { heatmap: string; overlay: string; description: string }
    }
    uncertainty_analysis: {
      uncertainty_metrics: {
        predictive_entropy: number
        normalized_entropy: number
        epistemic_uncertainty: number
        aleatoric_uncertainty: number
      }
      agreement: {
        score: number
        all_agree: boolean
        models_agreeing: number
      }
      confidence_intervals: Record<string, {
        mean: number
        std: number
        ci_95_lower: number
        ci_95_upper: number
      }>
      model_votes: Array<{
        model: string
        prediction: string
        confidence: number
      }>
      reliability_score: number
      interpretation: {
        confidence_level: string
        confidence_description: string
        agreement_description: string
        recommendation: string
      }
    }
    feature_importance: {
      regional_importance: Record<string, number>
      most_important_regions: string[]
      interpretation: string
    }
    explanation_summary: {
      headline: string
      confidence_assessment: string
      model_agreement: string
      clinical_recommendation: string
      technical_note: string
    }
  }
}

export async function analyzeWithExplainability(
  file: File,
  options: {
    useTta?: boolean
    includeExplainability?: boolean
    onProgress?: (progress: number) => void
  } = {}
): Promise<AdvancedAnalysisResponse> {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('use_tta', String(options.useTta ?? true))
  formData.append('include_explainability', String(options.includeExplainability ?? true))

  const response = await api.post<AdvancedAnalysisResponse>('/analyze', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: (progressEvent) => {
      if (progressEvent.total && options.onProgress) {
        const progress = Math.round((progressEvent.loaded * 50) / progressEvent.total)
        options.onProgress(progress)
      }
    },
  })

  return response.data
}

// Error handling helper
export function getErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    if (error.response) {
      // Server responded with error
      const data = error.response.data as { error?: string; message?: string }
      return data.error || data.message || `Server error: ${error.response.status}`
    } else if (error.request) {
      // Request made but no response
      return 'Unable to connect to server. Please check if the backend is running.'
    }
  }
  
  if (error instanceof Error) {
    return error.message
  }
  
  return 'An unexpected error occurred'
}
