import { motion } from 'framer-motion'
import { Card, Badge, ProgressBar } from '@/components/ui'
import { MODEL_INFO } from '@/utils/constants'
import { formatPercentage } from '@/utils/helpers'

interface ModelCardProps {
  name: string
  architecture: string
  accuracy: number
  f1Score: number
  weight: number
  parameters: string
  description: string
  delay?: number
}

export function ModelCard({
  name,
  architecture,
  accuracy,
  f1Score,
  weight,
  parameters,
  description,
  delay = 0
}: ModelCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5, delay }}
    >
      <Card variant="hover" className="h-full">
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div>
            <h3 className="text-xl font-bold text-white">{name}</h3>
            <p className="text-sm text-dark-400">{architecture}</p>
          </div>
          <Badge variant="primary">
            {(weight * 100).toFixed(0)}% Weight
          </Badge>
        </div>

        {/* Description */}
        <p className="text-sm text-dark-300 mb-6">
          {description}
        </p>

        {/* Metrics */}
        <div className="space-y-4">
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-dark-400">Accuracy</span>
              <span className="text-white font-medium">{formatPercentage(accuracy)}</span>
            </div>
            <ProgressBar 
              value={accuracy * 100} 
              size="sm" 
              variant="success" 
            />
          </div>
          
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-dark-400">F1 Score</span>
              <span className="text-white font-medium">{formatPercentage(f1Score)}</span>
            </div>
            <ProgressBar 
              value={f1Score * 100} 
              size="sm" 
              variant="default" 
            />
          </div>
        </div>

        {/* Footer stats */}
        <div className="mt-6 pt-4 border-t border-dark-700 flex justify-between text-sm">
          <span className="text-dark-500">Parameters</span>
          <span className="text-dark-300 font-mono">{parameters}</span>
        </div>
      </Card>
    </motion.div>
  )
}

// Model grid component
export function ModelGrid() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      {MODEL_INFO.map((model, index) => (
        <ModelCard
          key={model.name}
          {...model}
          delay={index * 0.15}
        />
      ))}
    </div>
  )
}

// Model comparison table
export function ModelComparisonTable() {
  return (
    <div className="overflow-x-auto">
      <table 
        className="w-full text-sm"
        role="table"
        aria-label="Model comparison"
      >
        <thead>
          <tr className="border-b border-dark-700">
            <th className="text-left py-3 px-4 text-dark-400 font-medium">Model</th>
            <th className="text-left py-3 px-4 text-dark-400 font-medium">Architecture</th>
            <th className="text-right py-3 px-4 text-dark-400 font-medium">Accuracy</th>
            <th className="text-right py-3 px-4 text-dark-400 font-medium">F1 Score</th>
            <th className="text-right py-3 px-4 text-dark-400 font-medium">Weight</th>
            <th className="text-right py-3 px-4 text-dark-400 font-medium">Params</th>
          </tr>
        </thead>
        <tbody>
          {MODEL_INFO.map((model) => (
            <motion.tr
              key={model.name}
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              className="border-b border-dark-800 hover:bg-dark-800/50 transition-colors"
            >
              <td className="py-4 px-4 font-medium text-white">{model.name}</td>
              <td className="py-4 px-4 text-dark-300">{model.architecture}</td>
              <td className="py-4 px-4 text-right">
                <span className="text-medical-green font-mono">
                  {formatPercentage(model.accuracy)}
                </span>
              </td>
              <td className="py-4 px-4 text-right">
                <span className="text-primary-400 font-mono">
                  {formatPercentage(model.f1Score)}
                </span>
              </td>
              <td className="py-4 px-4 text-right">
                <Badge size="sm" variant="primary">
                  {(model.weight * 100).toFixed(0)}%
                </Badge>
              </td>
              <td className="py-4 px-4 text-right text-dark-400 font-mono">
                {model.parameters}
              </td>
            </motion.tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
