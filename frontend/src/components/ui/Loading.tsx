import { cn } from '@/utils/helpers'

// Loading spinner with sizes
interface SpinnerProps {
  size?: 'sm' | 'md' | 'lg' | 'xl'
  className?: string
}

export function Spinner({ size = 'md', className }: SpinnerProps) {
  const sizes = {
    sm: 'w-4 h-4',
    md: 'w-8 h-8',
    lg: 'w-12 h-12',
    xl: 'w-16 h-16'
  }

  return (
    <svg
      className={cn('animate-spin text-primary-500', sizes[size], className)}
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
      role="status"
      aria-label="Loading"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  )
}

// Skeleton loading placeholder
interface SkeletonProps {
  className?: string
  variant?: 'text' | 'circular' | 'rectangular'
  width?: string | number
  height?: string | number
}

export function Skeleton({ 
  className, 
  variant = 'text',
  width,
  height 
}: SkeletonProps) {
  const variants = {
    text: 'h-4 rounded',
    circular: 'rounded-full',
    rectangular: 'rounded-lg'
  }

  return (
    <div
      className={cn(
        'bg-dark-700 animate-pulse',
        variants[variant],
        className
      )}
      style={{ width, height }}
      aria-hidden="true"
    />
  )
}

// Full page loading state
interface LoadingOverlayProps {
  message?: string
  progress?: number
}

export function LoadingOverlay({ message = 'Loading...', progress }: LoadingOverlayProps) {
  return (
    <div 
      className="fixed inset-0 z-50 flex items-center justify-center bg-dark-950/80 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-label={message}
    >
      <div className="text-center">
        <Spinner size="xl" className="mx-auto mb-4" />
        <p className="text-lg font-medium text-white">{message}</p>
        {progress !== undefined && (
          <div className="mt-4 w-48 mx-auto">
            <div className="h-2 bg-dark-800 rounded-full overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-primary-600 to-primary-400 transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
            <p className="mt-2 text-sm text-dark-400">{Math.round(progress)}%</p>
          </div>
        )}
      </div>
    </div>
  )
}

// Analysis loading state with brain animation
export function AnalysisLoader({ progress = 0 }: { progress?: number }) {
  return (
    <div className="flex flex-col items-center justify-center py-12">
      {/* Animated brain icon */}
      <div className="relative mb-6">
        <div className="w-24 h-24 rounded-full bg-gradient-to-br from-primary-600/20 to-primary-400/20 flex items-center justify-center">
          <svg 
            className="w-12 h-12 text-primary-400" 
            viewBox="0 0 24 24" 
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path 
              d="M12 3C7.03 3 3 7.03 3 12C3 16.97 7.03 21 12 21C16.97 21 21 16.97 21 12" 
              stroke="currentColor" 
              strokeWidth="2" 
              strokeLinecap="round"
              className="animate-spin origin-center"
              style={{ animationDuration: '2s' }}
            />
            <circle cx="12" cy="12" r="4" fill="currentColor" opacity="0.3" />
            <path 
              d="M12 8V12L14 14" 
              stroke="currentColor" 
              strokeWidth="2" 
              strokeLinecap="round"
            />
          </svg>
        </div>
        {/* Pulsing rings */}
        <div className="absolute inset-0 rounded-full border-2 border-primary-500/30 animate-ping" />
        <div className="absolute inset-0 rounded-full border border-primary-500/20 animate-pulse" />
      </div>

      {/* Progress */}
      <div className="w-64">
        <div className="flex justify-between text-sm mb-2">
          <span className="text-dark-400">Analyzing image...</span>
          <span className="text-primary-400 font-medium">{Math.round(progress)}%</span>
        </div>
        <div className="h-2 bg-dark-800 rounded-full overflow-hidden">
          <div 
            className="h-full bg-gradient-to-r from-primary-600 to-primary-400 transition-all duration-500 ease-out"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Status steps */}
      <div className="mt-6 text-sm text-dark-400">
        {progress < 25 && 'Preprocessing image...'}
        {progress >= 25 && progress < 50 && 'Running neural networks...'}
        {progress >= 50 && progress < 75 && 'Applying test-time augmentation...'}
        {progress >= 75 && progress < 100 && 'Generating predictions...'}
        {progress >= 100 && 'Complete!'}
      </div>
    </div>
  )
}
