import type { TumorClassInfo, ModelInfo } from '@/types'

export const TUMOR_CLASSES: Record<string, TumorClassInfo> = {
  glioma: {
    name: 'glioma',
    displayName: 'Glioma',
    description: 'A tumor that originates in the glial cells of the brain. Can range from low-grade (slow-growing) to high-grade (aggressive, such as glioblastoma).',
    severity: 'high',
    color: '#ef4444',
    icon: '🔴',
    recommendations: [
      'Immediate consultation with a neuro-oncologist recommended',
      'Additional imaging (MRI with contrast) may be needed',
      'Biopsy may be required for definitive diagnosis',
      'Discuss treatment options including surgery, radiation, and chemotherapy'
    ]
  },
  meningioma: {
    name: 'meningioma',
    displayName: 'Meningioma',
    description: 'A tumor arising from the meninges, the membranes surrounding the brain. Most are benign and slow-growing.',
    severity: 'moderate',
    color: '#eab308',
    icon: '🟡',
    recommendations: [
      'Follow-up imaging recommended in 3-6 months',
      'Consultation with neurosurgeon if symptomatic',
      'Many cases can be monitored without immediate intervention',
      'Consider watchful waiting approach for small tumors'
    ]
  },
  pituitary: {
    name: 'pituitary',
    displayName: 'Pituitary Tumor',
    description: 'A growth in the pituitary gland at the base of the brain. Usually benign, but can affect hormone production.',
    severity: 'moderate',
    color: '#f97316',
    icon: '🟠',
    recommendations: [
      'Endocrine evaluation recommended',
      'Hormone level testing advised',
      'Visual field testing may be needed',
      'Treatment depends on tumor type and hormone involvement'
    ]
  },
  notumor: {
    name: 'notumor',
    displayName: 'No Tumor Detected',
    description: 'The scan appears normal with no detectable tumor masses. Brain tissue shows typical characteristics.',
    severity: 'normal',
    color: '#22c55e',
    icon: '🟢',
    recommendations: [
      'No immediate action required',
      'Continue routine health monitoring',
      'Consult physician if symptoms persist',
      'Maintain regular check-up schedule'
    ]
  }
}

export const MODEL_INFO: ModelInfo[] = [
  {
    name: 'EfficientNet-B3',
    architecture: 'EfficientNet',
    accuracy: 0.954,
    f1Score: 0.952,
    weight: 0.40,
    parameters: '12M',
    description: 'Highest performing model using compound scaling. Excellent at capturing fine-grained tumor features with efficient computation.'
  },
  {
    name: 'ResNet-50',
    architecture: 'ResNet',
    accuracy: 0.938,
    f1Score: 0.935,
    weight: 0.35,
    parameters: '25.6M',
    description: 'Deep residual network with skip connections. Robust feature extraction and proven performance on medical imaging tasks.'
  },
  {
    name: 'DenseNet-121',
    architecture: 'DenseNet',
    accuracy: 0.925,
    f1Score: 0.921,
    weight: 0.25,
    parameters: '8M',
    description: 'Densely connected network for maximum feature reuse. Excellent gradient flow and compact representation learning.'
  }
]

export const FEATURES = [
  {
    title: 'Weighted Ensemble',
    description: 'Three state-of-the-art CNNs combined with F1-optimized voting weights for superior accuracy.',
    iconName: 'bolt',
    color: 'blue' as const
  },
  {
    title: 'Test-Time Augmentation',
    description: '7 augmentations per model for robust predictions even on images from unknown sources.',
    iconName: 'refresh',
    color: 'purple' as const
  },
  {
    title: 'Calibrated Confidence',
    description: 'Temperature-scaled probabilities for reliable uncertainty estimation in clinical settings.',
    iconName: 'chart',
    color: 'green' as const
  },
  {
    title: 'Grad-CAM Explainability',
    description: 'Visual attention maps showing exactly what the model focuses on for each prediction.',
    iconName: 'eye',
    color: 'orange' as const
  }
]

export const STATS = [
  { label: 'Neural Networks', value: '3', description: 'Ensemble models' },
  { label: 'TTA Transforms', value: '7', description: 'Per prediction' },
  { label: 'Accuracy', value: '95.4%', description: 'Lead model' },
  { label: 'Processing', value: '<2s', description: 'Per image' }
]

export const NAV_ITEMS = [
  { label: 'Home', href: '/' },
  { label: 'Analyze', href: '/analyze' },
  { label: 'Models', href: '/models' },
  { label: 'About', href: '/about' }
]

export const KEYBOARD_SHORTCUTS = {
  ANALYZE: 'a',
  HOME: 'h',
  MODELS: 'm',
  ABOUT: 'b',
  UPLOAD: 'u',
  ESCAPE: 'Escape'
}
