import { motion } from 'framer-motion'
import { 
  CheckCircleIcon, 
  ExclamationTriangleIcon,
  InformationCircleIcon,
  ChartBarIcon,
  EyeIcon,
  BeakerIcon,
  ClockIcon
} from '@heroicons/react/24/outline'
import { Card, Badge, CircularProgress } from '@/components/ui'
import { cn, formatPercentage, getSeverityBg } from '@/utils/helpers'
import { TUMOR_CLASSES } from '@/utils/constants'
import type { PredictionResult } from '@/types'

interface ResultsDisplayProps {
  result: PredictionResult
  showGradcam?: boolean
}

export function ResultsDisplay({ result, showGradcam = true }: ResultsDisplayProps) {
  const { prediction, probabilities, model_votes, ensemble_stats, gradcam, processing_time } = result
  const classInfo = TUMOR_CLASSES[prediction.class]

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="space-y-6"
    >
      {/* Main Result Card */}
      <Card 
        className={cn(
          'relative overflow-hidden',
          getSeverityBg(prediction.severity)
        )}
      >
        {/* Decorative gradient */}
        <div 
          className="absolute top-0 right-0 w-64 h-64 opacity-10 blur-3xl"
          style={{ background: classInfo?.color || '#6366f1' }}
        />
        
        <div className="relative flex flex-col md:flex-row md:items-center gap-6">
          {/* Confidence circle */}
          <div className="flex-shrink-0">
            <CircularProgress
              value={prediction.confidence * 100}
              size={140}
              variant={prediction.severity === 'normal' ? 'success' : prediction.severity === 'high' ? 'danger' : 'warning'}
            />
          </div>
          
          {/* Result details */}
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-3">
              <span className="text-3xl" role="img" aria-label={prediction.display_name}>
                {classInfo?.icon}
              </span>
              <div>
                <h2 className="text-2xl font-bold text-white">
                  {prediction.display_name}
                </h2>
                <Badge 
                  variant={prediction.severity === 'normal' ? 'success' : prediction.severity === 'high' ? 'danger' : 'warning'}
                  className="mt-1"
                >
                  {prediction.severity === 'normal' ? 'Normal' : prediction.severity === 'high' ? 'High Priority' : 'Moderate Priority'}
                </Badge>
              </div>
            </div>
            
            <p className="text-dark-300 leading-relaxed">
              {prediction.description}
            </p>

            {/* Quick stats */}
            <div className="flex flex-wrap gap-4 mt-4 pt-4 border-t border-dark-700/50">
              <div className="flex items-center gap-2 text-sm text-dark-400">
                <BeakerIcon className="w-4 h-4" />
                <span>{ensemble_stats.models_used} Models</span>
              </div>
              <div className="flex items-center gap-2 text-sm text-dark-400">
                <ChartBarIcon className="w-4 h-4" />
                <span>{ensemble_stats.tta_transforms}× TTA</span>
              </div>
              <div className="flex items-center gap-2 text-sm text-dark-400">
                <ClockIcon className="w-4 h-4" />
                <span>{processing_time.toFixed(2)}s</span>
              </div>
            </div>
          </div>
        </div>
      </Card>

      {/* Probability Distribution */}
      <Card>
        <div className="flex items-center gap-2 mb-4">
          <ChartBarIcon className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Class Probabilities</h3>
        </div>
        
        <div className="space-y-4">
          {probabilities
            .sort((a, b) => b.probability - a.probability)
            .map((prob, index) => {
              const info = TUMOR_CLASSES[prob.class]
              const isTop = index === 0
              
              return (
                <motion.div
                  key={prob.class}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <div className="flex items-center gap-2">
                      <span className="text-lg">{info?.icon}</span>
                      <span className={cn(
                        'text-sm font-medium',
                        isTop ? 'text-white' : 'text-dark-300'
                      )}>
                        {info?.displayName || prob.class}
                      </span>
                    </div>
                    <span className={cn(
                      'text-sm font-mono',
                      isTop ? 'text-white' : 'text-dark-400'
                    )}>
                      {formatPercentage(prob.probability)}
                    </span>
                  </div>
                  <div className="h-2 bg-dark-700 rounded-full overflow-hidden">
                    <motion.div
                      className="h-full rounded-full"
                      style={{ backgroundColor: info?.color || '#6366f1' }}
                      initial={{ width: 0 }}
                      animate={{ width: `${prob.probability * 100}%` }}
                      transition={{ duration: 0.5, delay: index * 0.1 }}
                    />
                  </div>
                </motion.div>
              )
            })}
        </div>
      </Card>

      {/* Model Votes */}
      <Card>
        <div className="flex items-center gap-2 mb-4">
          <BeakerIcon className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Ensemble Breakdown</h3>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {model_votes.map((vote, index) => {
            const voteInfo = TUMOR_CLASSES[vote.prediction]
            return (
              <motion.div
                key={vote.model}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className="p-4 rounded-xl bg-dark-800/50 border border-dark-700"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-white">{vote.model}</span>
                  <Badge size="sm" variant="primary">
                    Weight: {(vote.weight * 100).toFixed(0)}%
                  </Badge>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xl">{voteInfo?.icon}</span>
                  <div>
                    <p className="text-sm text-dark-300">{voteInfo?.displayName}</p>
                    <p className="text-xs text-dark-500">
                      {formatPercentage(vote.confidence)} confidence
                    </p>
                  </div>
                </div>
              </motion.div>
            )
          })}
        </div>

        {/* Agreement score */}
        <div className="mt-4 pt-4 border-t border-dark-700">
          <div className="flex items-center justify-between">
            <span className="text-sm text-dark-400">Model Agreement</span>
            <span className={cn(
              'text-sm font-medium',
              ensemble_stats.agreement_score >= 0.8 ? 'text-medical-green' : 
              ensemble_stats.agreement_score >= 0.6 ? 'text-medical-yellow' : 'text-medical-red'
            )}>
              {formatPercentage(ensemble_stats.agreement_score)}
            </span>
          </div>
        </div>
      </Card>

      {/* Grad-CAM Visualization */}
      {showGradcam && gradcam && (
        <Card>
          <div className="flex items-center gap-2 mb-4">
            <EyeIcon className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Attention Visualization</h3>
          </div>
          
          <p className="text-sm text-dark-400 mb-4">
            Grad-CAM heatmap showing regions the model focused on for classification.
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <p className="text-xs text-dark-500 mb-2 text-center">Heatmap</p>
              <div className="rounded-xl overflow-hidden bg-dark-800">
                <img 
                  src={`data:image/png;base64,${gradcam.heatmap}`}
                  alt="Grad-CAM attention heatmap"
                  className="w-full h-auto"
                />
              </div>
            </div>
            <div>
              <p className="text-xs text-dark-500 mb-2 text-center">Overlay</p>
              <div className="rounded-xl overflow-hidden bg-dark-800">
                <img 
                  src={`data:image/png;base64,${gradcam.overlay}`}
                  alt="Grad-CAM overlay on original image"
                  className="w-full h-auto"
                />
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Recommendations */}
      <Card className={getSeverityBg(prediction.severity)}>
        <div className="flex items-center gap-2 mb-4">
          {prediction.severity === 'normal' ? (
            <CheckCircleIcon className="w-5 h-5 text-medical-green" />
          ) : prediction.severity === 'high' ? (
            <ExclamationTriangleIcon className="w-5 h-5 text-medical-red" />
          ) : (
            <InformationCircleIcon className="w-5 h-5 text-medical-yellow" />
          )}
          <h3 className="text-lg font-semibold text-white">Recommendations</h3>
        </div>
        
        <ul className="space-y-3" role="list">
          {prediction.recommendations.map((rec, index) => (
            <motion.li
              key={index}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className="flex items-start gap-3"
            >
              <span className={cn(
                'w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 text-xs font-medium',
                prediction.severity === 'normal' ? 'bg-medical-green/20 text-medical-green' :
                prediction.severity === 'high' ? 'bg-medical-red/20 text-medical-red' :
                'bg-medical-yellow/20 text-medical-yellow'
              )}>
                {index + 1}
              </span>
              <span className="text-dark-300">{rec}</span>
            </motion.li>
          ))}
        </ul>

        {/* Disclaimer */}
        <div className="mt-6 pt-4 border-t border-dark-700/50">
          <p className="text-xs text-dark-500">
            ⚠️ <strong>Disclaimer:</strong> This AI analysis is for research purposes only and should not 
            replace professional medical diagnosis. Always consult with qualified healthcare providers 
            for medical decisions.
          </p>
        </div>
      </Card>
    </motion.div>
  )
}
