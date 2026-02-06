import { cn } from '@/utils/helpers'
import { motion } from 'framer-motion'

interface ProgressBarProps {
  value: number
  max?: number
  size?: 'sm' | 'md' | 'lg'
  variant?: 'default' | 'success' | 'warning' | 'danger'
  showLabel?: boolean
  label?: string
  animated?: boolean
  className?: string
}

export function ProgressBar({
  value,
  max = 100,
  size = 'md',
  variant = 'default',
  showLabel = false,
  label,
  animated = true,
  className
}: ProgressBarProps) {
  const percentage = Math.min(Math.max((value / max) * 100, 0), 100)

  const sizes = {
    sm: 'h-1.5',
    md: 'h-2.5',
    lg: 'h-4'
  }

  const variants = {
    default: 'from-primary-600 to-primary-400',
    success: 'from-medical-green to-green-400',
    warning: 'from-medical-yellow to-yellow-400',
    danger: 'from-medical-red to-red-400'
  }

  return (
    <div className={cn('w-full', className)}>
      {(showLabel || label) && (
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm text-dark-300">{label || 'Progress'}</span>
          <span className="text-sm font-medium text-dark-100">
            {Math.round(percentage)}%
          </span>
        </div>
      )}
      <div
        className={cn(
          'w-full bg-dark-800 rounded-full overflow-hidden',
          sizes[size]
        )}
        role="progressbar"
        aria-valuenow={value}
        aria-valuemin={0}
        aria-valuemax={max}
        aria-label={label || `Progress: ${Math.round(percentage)}%`}
      >
        <motion.div
          className={cn(
            'h-full rounded-full bg-gradient-to-r',
            variants[variant]
          )}
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ 
            duration: animated ? 0.5 : 0, 
            ease: 'easeOut' 
          }}
        />
      </div>
    </div>
  )
}

// Circular progress indicator
interface CircularProgressProps {
  value: number
  size?: number
  strokeWidth?: number
  variant?: 'default' | 'success' | 'warning' | 'danger'
  showValue?: boolean
  className?: string
}

export function CircularProgress({
  value,
  size = 120,
  strokeWidth = 8,
  variant = 'default',
  showValue = true,
  className
}: CircularProgressProps) {
  const percentage = Math.min(Math.max(value, 0), 100)
  const radius = (size - strokeWidth) / 2
  const circumference = radius * 2 * Math.PI
  const offset = circumference - (percentage / 100) * circumference

  const colors = {
    default: 'stroke-primary-500',
    success: 'stroke-medical-green',
    warning: 'stroke-medical-yellow',
    danger: 'stroke-medical-red'
  }

  return (
    <div 
      className={cn('relative inline-flex items-center justify-center', className)}
      role="progressbar"
      aria-valuenow={percentage}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <svg width={size} height={size} className="transform -rotate-90">
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          strokeWidth={strokeWidth}
          fill="none"
          className="stroke-dark-700"
        />
        {/* Progress circle */}
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          strokeWidth={strokeWidth}
          fill="none"
          className={colors[variant]}
          strokeLinecap="round"
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 1, ease: 'easeOut' }}
          style={{
            strokeDasharray: circumference
          }}
        />
      </svg>
      {showValue && (
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-2xl font-bold text-white">
            {Math.round(percentage)}%
          </span>
        </div>
      )}
    </div>
  )
}
