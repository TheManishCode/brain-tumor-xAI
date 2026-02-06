import { motion, AnimatePresence } from 'framer-motion'
import { useState } from 'react'
import {
  EyeIcon,
  ChartBarIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  BeakerIcon,
  SparklesIcon
} from '@heroicons/react/24/outline'
import { Card, Badge } from '@/components/ui'
import { cn } from '@/utils/helpers'

interface VisualizationData {
  heatmap: string
  overlay: string
  description: string
}

interface UncertaintyMetrics {
  predictive_entropy: number
  normalized_entropy: number
  epistemic_uncertainty: number
  aleatoric_uncertainty: number
}

interface ConfidenceInterval {
  mean: number
  std: number
  ci_95_lower: number
  ci_95_upper: number
}

interface ModelVote {
  model: string
  prediction: string
  confidence: number
  all_probabilities?: Record<string, number>
}

interface Interpretation {
  confidence_level: string
  confidence_description: string
  agreement_description: string
  recommendation: string
}

interface ExplainabilityData {
  predicted_class: string
  confidence: number
  visualizations: {
    gradcam?: VisualizationData
    gradcam_pp?: VisualizationData
    integrated_gradients?: VisualizationData
    saliency?: VisualizationData
    lime?: VisualizationData & {
      metadata?: {
        num_segments: number
        model_score: number
        top_features: number[]
        bottom_features: number[]
      }
    }
  }
  uncertainty_analysis: {
    uncertainty_metrics: UncertaintyMetrics
    agreement: {
      score: number
      all_agree: boolean
      models_agreeing: number
    }
    confidence_intervals: Record<string, ConfidenceInterval>
    model_votes: ModelVote[]
    reliability_score: number
    interpretation: Interpretation
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

interface ExplainabilityPanelProps {
  data: ExplainabilityData
  className?: string
}

export function ExplainabilityPanel({ data, className }: ExplainabilityPanelProps) {
  const [expandedSection, setExpandedSection] = useState<string | null>('visualizations')
  const [selectedViz, setSelectedViz] = useState<string>('gradcam')

  // Safety check for missing data
  if (!data || !data.visualizations || !data.uncertainty_analysis || !data.explanation_summary) {
    return (
      <div className={cn('p-6 rounded-xl bg-dark-800/50 border border-dark-700', className)}>
        <div className="flex items-center gap-3 text-dark-400">
          <ExclamationTriangleIcon className="w-5 h-5 text-yellow-500" />
          <span>Explainability data is incomplete or unavailable</span>
        </div>
      </div>
    )
  }

  const toggleSection = (section: string) => {
    setExpandedSection(expandedSection === section ? null : section)
  }

  const vizLabels: Record<string, string> = {
    'gradcam': 'Grad-CAM',
    'gradcam_pp': 'Grad-CAM++',
    'integrated_gradients': 'Integrated Gradients',
    'saliency': 'Saliency',
    'lime': 'LIME'
  }

  const vizOptions = Object.entries(data.visualizations || {}).filter(([, value]) => value != null).map(([key, value]) => ({
    key,
    label: vizLabels[key] || key.replace('_', ' '),
    data: value
  }))

  const currentViz = data.visualizations?.[selectedViz as keyof typeof data.visualizations]

  return (
    <div className={cn('space-y-4', className)}>
      {/* Summary Header */}
      <Card className="bg-gradient-to-br from-primary-900/50 to-dark-900 border-primary-700/50">
        <div className="flex items-start gap-4">
          <div className="p-3 rounded-xl bg-primary-600/20">
            <SparklesIcon className="w-6 h-6 text-primary-400" />
          </div>
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-white mb-1">
              {data.explanation_summary.headline}
            </h3>
            <p className="text-sm text-dark-300">
              {data.explanation_summary.confidence_assessment}
            </p>
            <div className="flex items-center gap-2 mt-3">
              <Badge 
                variant={data.uncertainty_analysis.reliability_score > 0.7 ? 'success' : 
                        data.uncertainty_analysis.reliability_score > 0.4 ? 'warning' : 'danger'}
              >
                Reliability: {(data.uncertainty_analysis.reliability_score * 100).toFixed(0)}%
              </Badge>
              <Badge variant="primary">
                {data.uncertainty_analysis.interpretation.confidence_level} Confidence
              </Badge>
            </div>
          </div>
        </div>
      </Card>

      {/* Visualizations Section */}
      <Card>
        <button
          onClick={() => toggleSection('visualizations')}
          className="w-full flex items-center justify-between p-1"
        >
          <div className="flex items-center gap-2">
            <EyeIcon className="w-5 h-5 text-primary-400" />
            <h4 className="font-semibold text-white">Attention Visualizations</h4>
          </div>
          {expandedSection === 'visualizations' ? (
            <ChevronUpIcon className="w-5 h-5 text-dark-400" />
          ) : (
            <ChevronDownIcon className="w-5 h-5 text-dark-400" />
          )}
        </button>

        <AnimatePresence>
          {expandedSection === 'visualizations' && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden"
            >
              <div className="mt-4 space-y-4">
                {/* Visualization Tabs */}
                <div className="flex flex-wrap gap-2">
                  {vizOptions.map(({ key, label }) => (
                    <button
                      key={key}
                      onClick={() => setSelectedViz(key)}
                      className={cn(
                        'px-3 py-1.5 text-sm rounded-lg transition-all',
                        selectedViz === key
                          ? 'bg-primary-600 text-white'
                          : 'bg-dark-700 text-dark-300 hover:bg-dark-600'
                      )}
                    >
                      {label}
                    </button>
                  ))}
                </div>

                {/* Current Visualization */}
                {currentViz && (
                  <div className="space-y-3">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-xs text-dark-500 mb-2 text-center">
                          {selectedViz === 'lime' ? 'Importance Map' : 'Attention Heatmap'}
                        </p>
                        <div className="rounded-xl overflow-hidden bg-dark-800 aspect-square">
                          <img
                            src={`data:image/png;base64,${currentViz.heatmap}`}
                            alt="Attention heatmap"
                            className="w-full h-full object-cover"
                          />
                        </div>
                      </div>
                      <div>
                        <p className="text-xs text-dark-500 mb-2 text-center">Overlay on Image</p>
                        <div className="rounded-xl overflow-hidden bg-dark-800 aspect-square">
                          <img
                            src={`data:image/png;base64,${currentViz.overlay}`}
                            alt="Heatmap overlay"
                            className="w-full h-full object-cover"
                          />
                        </div>
                      </div>
                    </div>
                    <p className="text-sm text-dark-400 text-center">
                      {currentViz.description}
                    </p>
                    
                    {/* LIME-specific metadata */}
                    {selectedViz === 'lime' && data.visualizations.lime?.metadata && (
                      <div className="mt-3 p-3 rounded-lg bg-dark-800/50 border border-dark-700">
                        <div className="flex items-center gap-2 mb-2">
                          <BeakerIcon className="w-4 h-4 text-cyan-400" />
                          <h6 className="text-xs font-medium text-cyan-400">LIME Analysis Details</h6>
                        </div>
                        <div className="grid grid-cols-2 gap-3 text-xs">
                          <div>
                            <span className="text-dark-500">Segments analyzed:</span>
                            <span className="ml-2 text-dark-300">{data.visualizations.lime.metadata.num_segments}</span>
                          </div>
                          <div>
                            <span className="text-dark-500">Model fit score:</span>
                            <span className="ml-2 text-dark-300">{(data.visualizations.lime.metadata.model_score * 100).toFixed(1)}%</span>
                          </div>
                        </div>
                        <div className="mt-2 text-xs text-dark-500">
                          <span className="text-green-400">Red regions</span> support the prediction, 
                          <span className="text-blue-400 ml-1">blue regions</span> oppose it.
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Feature Importance */}
                <div className="mt-4 p-4 rounded-xl bg-dark-800/50 border border-dark-700">
                  <h5 className="text-sm font-medium text-white mb-3">Regional Importance</h5>
                  <div className="space-y-2">
                    {Object.entries(data.feature_importance.regional_importance)
                      .sort(([, a], [, b]) => b - a)
                      .map(([region, importance]) => (
                        <div key={region} className="space-y-1">
                          <div className="flex justify-between text-xs">
                            <span className="text-dark-300 capitalize">
                              {region.replace('_', ' ')}
                            </span>
                            <span className="text-dark-400">
                              {(importance * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="h-1.5 bg-dark-700 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-gradient-to-r from-primary-600 to-primary-400 rounded-full"
                              style={{ width: `${importance * 100}%` }}
                            />
                          </div>
                        </div>
                      ))}
                  </div>
                  <p className="mt-3 text-xs text-dark-500">
                    {data.feature_importance.interpretation}
                  </p>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </Card>

      {/* Uncertainty Analysis Section */}
      <Card>
        <button
          onClick={() => toggleSection('uncertainty')}
          className="w-full flex items-center justify-between p-1"
        >
          <div className="flex items-center gap-2">
            <ChartBarIcon className="w-5 h-5 text-primary-400" />
            <h4 className="font-semibold text-white">Uncertainty Analysis</h4>
          </div>
          {expandedSection === 'uncertainty' ? (
            <ChevronUpIcon className="w-5 h-5 text-dark-400" />
          ) : (
            <ChevronDownIcon className="w-5 h-5 text-dark-400" />
          )}
        </button>

        <AnimatePresence>
          {expandedSection === 'uncertainty' && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden"
            >
              <div className="mt-4 space-y-4">
                {/* Uncertainty Metrics */}
                <div className="grid grid-cols-2 gap-3">
                  <MetricCard
                    label="Epistemic Uncertainty"
                    value={data.uncertainty_analysis.uncertainty_metrics.epistemic_uncertainty}
                    description="Model uncertainty (can improve with more data)"
                    color="blue"
                  />
                  <MetricCard
                    label="Aleatoric Uncertainty"
                    value={data.uncertainty_analysis.uncertainty_metrics.aleatoric_uncertainty}
                    description="Data uncertainty (inherent noise)"
                    color="purple"
                  />
                  <MetricCard
                    label="Model Agreement"
                    value={data.uncertainty_analysis.agreement.score}
                    description={`${data.uncertainty_analysis.agreement.models_agreeing} of 3 models agree`}
                    color="green"
                  />
                  <MetricCard
                    label="Predictive Entropy"
                    value={data.uncertainty_analysis.uncertainty_metrics.normalized_entropy}
                    description="Overall prediction uncertainty"
                    color="orange"
                  />
                </div>

                {/* Model Votes */}
                <div className="p-4 rounded-xl bg-dark-800/50 border border-dark-700">
                  <h5 className="text-sm font-medium text-white mb-3 flex items-center gap-2">
                    <BeakerIcon className="w-4 h-4" />
                    Ensemble Model Votes
                  </h5>
                  <div className="space-y-3">
                    {data.uncertainty_analysis.model_votes.map((vote) => (
                      <div
                        key={vote.model}
                        className="flex items-center justify-between p-3 rounded-lg bg-dark-700/50"
                      >
                        <div>
                          <p className="text-sm font-medium text-white">
                            {vote.model.replace('_', '-')}
                          </p>
                          <p className="text-xs text-dark-400">
                            Predicts: {vote.prediction}
                          </p>
                        </div>
                        <Badge variant="primary">
                          {(vote.confidence * 100).toFixed(1)}%
                        </Badge>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Interpretation */}
                <div className={cn(
                  'p-4 rounded-xl border',
                  data.uncertainty_analysis.reliability_score > 0.7
                    ? 'bg-medical-green/10 border-medical-green/30'
                    : data.uncertainty_analysis.reliability_score > 0.4
                    ? 'bg-medical-yellow/10 border-medical-yellow/30'
                    : 'bg-medical-red/10 border-medical-red/30'
                )}>
                  <div className="flex items-start gap-3">
                    {data.uncertainty_analysis.reliability_score > 0.7 ? (
                      <InformationCircleIcon className="w-5 h-5 text-medical-green flex-shrink-0" />
                    ) : (
                      <ExclamationTriangleIcon className={cn(
                        'w-5 h-5 flex-shrink-0',
                        data.uncertainty_analysis.reliability_score > 0.4
                          ? 'text-medical-yellow'
                          : 'text-medical-red'
                      )} />
                    )}
                    <div>
                      <p className="text-sm font-medium text-white">
                        {data.uncertainty_analysis.interpretation.confidence_level} Confidence
                      </p>
                      <p className="text-xs text-dark-300 mt-1">
                        {data.uncertainty_analysis.interpretation.recommendation}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </Card>

      {/* Confidence Intervals Section */}
      <Card>
        <button
          onClick={() => toggleSection('confidence')}
          className="w-full flex items-center justify-between p-1"
        >
          <div className="flex items-center gap-2">
            <InformationCircleIcon className="w-5 h-5 text-primary-400" />
            <h4 className="font-semibold text-white">Class Confidence Intervals</h4>
          </div>
          {expandedSection === 'confidence' ? (
            <ChevronUpIcon className="w-5 h-5 text-dark-400" />
          ) : (
            <ChevronDownIcon className="w-5 h-5 text-dark-400" />
          )}
        </button>

        <AnimatePresence>
          {expandedSection === 'confidence' && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden"
            >
              <div className="mt-4 space-y-4">
                {Object.entries(data.uncertainty_analysis.confidence_intervals)
                  .sort(([, a], [, b]) => b.mean - a.mean)
                  .map(([className, ci]) => (
                    <div key={className} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium text-white capitalize">
                          {className.replace('_', ' ')}
                        </span>
                        <span className="text-sm text-dark-400">
                          {(ci.mean * 100).toFixed(1)}% ± {(ci.std * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="relative h-3 bg-dark-700 rounded-full overflow-hidden">
                        {/* CI Range */}
                        <div
                          className="absolute h-full bg-primary-700/50 rounded-full"
                          style={{
                            left: `${ci.ci_95_lower * 100}%`,
                            width: `${(ci.ci_95_upper - ci.ci_95_lower) * 100}%`
                          }}
                        />
                        {/* Mean Point */}
                        <div
                          className="absolute w-2 h-full bg-primary-400 rounded-full"
                          style={{ left: `calc(${ci.mean * 100}% - 4px)` }}
                        />
                      </div>
                      <div className="flex justify-between text-xs text-dark-500">
                        <span>95% CI: {(ci.ci_95_lower * 100).toFixed(1)}%</span>
                        <span>{(ci.ci_95_upper * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  ))}
                <p className="text-xs text-dark-500 text-center mt-4">
                  Confidence intervals show the range of predictions across ensemble models
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </Card>

      {/* Technical Note */}
      <p className="text-xs text-dark-500 text-center px-4">
        {data.explanation_summary.technical_note}
      </p>
    </div>
  )
}

function MetricCard({
  label,
  value,
  description,
  color
}: {
  label: string
  value: number
  description: string
  color: 'blue' | 'purple' | 'green' | 'orange'
}) {
  const colorClasses = {
    blue: 'from-blue-600/20 to-blue-400/20 border-blue-700/50',
    purple: 'from-purple-600/20 to-purple-400/20 border-purple-700/50',
    green: 'from-green-600/20 to-green-400/20 border-green-700/50',
    orange: 'from-orange-600/20 to-orange-400/20 border-orange-700/50'
  }

  return (
    <div className={cn(
      'p-3 rounded-xl bg-gradient-to-br border',
      colorClasses[color]
    )}>
      <p className="text-xs text-dark-400 mb-1">{label}</p>
      <p className="text-lg font-bold text-white">
        {(value * 100).toFixed(1)}%
      </p>
      <p className="text-xs text-dark-500 mt-1">{description}</p>
    </div>
  )
}

export default ExplainabilityPanel
