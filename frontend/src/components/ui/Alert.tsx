import {
  CheckCircleIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  XCircleIcon,
  XMarkIcon
} from '@heroicons/react/24/outline'
import { cn } from '@/utils/helpers'
import { motion, AnimatePresence } from 'framer-motion'

interface AlertProps {
  variant?: 'info' | 'success' | 'warning' | 'error'
  title?: string
  children: React.ReactNode
  dismissible?: boolean
  onDismiss?: () => void
  className?: string
  icon?: React.ReactNode
}

export function Alert({
  variant = 'info',
  title,
  children,
  dismissible = false,
  onDismiss,
  className,
  icon
}: AlertProps) {
  const variants = {
    info: {
      container: 'bg-medical-blue/10 border-medical-blue/20 text-medical-blue',
      icon: <InformationCircleIcon className="w-5 h-5" />
    },
    success: {
      container: 'bg-medical-green/10 border-medical-green/20 text-medical-green',
      icon: <CheckCircleIcon className="w-5 h-5" />
    },
    warning: {
      container: 'bg-medical-yellow/10 border-medical-yellow/20 text-medical-yellow',
      icon: <ExclamationTriangleIcon className="w-5 h-5" />
    },
    error: {
      container: 'bg-medical-red/10 border-medical-red/20 text-medical-red',
      icon: <XCircleIcon className="w-5 h-5" />
    }
  }

  const config = variants[variant]

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      role="alert"
      className={cn(
        'relative flex gap-3 p-4 rounded-xl border',
        config.container,
        className
      )}
    >
      <div className="flex-shrink-0">{icon || config.icon}</div>
      <div className="flex-1 min-w-0">
        {title && (
          <h4 className="font-semibold mb-1">{title}</h4>
        )}
        <div className="text-sm opacity-90">{children}</div>
      </div>
      {dismissible && onDismiss && (
        <button
          onClick={onDismiss}
          className="flex-shrink-0 p-1 rounded hover:bg-white/10 transition-colors"
          aria-label="Dismiss alert"
        >
          <XMarkIcon className="w-4 h-4" />
        </button>
      )}
    </motion.div>
  )
}

// Inline status message
interface StatusMessageProps {
  type: 'success' | 'error' | 'warning' | 'info'
  message: string
  className?: string
}

export function StatusMessage({ type, message, className }: StatusMessageProps) {
  const icons = {
    success: <CheckCircleIcon className="w-4 h-4 text-medical-green" />,
    error: <XCircleIcon className="w-4 h-4 text-medical-red" />,
    warning: <ExclamationTriangleIcon className="w-4 h-4 text-medical-yellow" />,
    info: <InformationCircleIcon className="w-4 h-4 text-medical-blue" />
  }

  const colors = {
    success: 'text-medical-green',
    error: 'text-medical-red',
    warning: 'text-medical-yellow',
    info: 'text-medical-blue'
  }

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, height: 0 }}
        animate={{ opacity: 1, height: 'auto' }}
        exit={{ opacity: 0, height: 0 }}
        className={cn('flex items-center gap-2 text-sm', colors[type], className)}
      >
        {icons[type]}
        <span>{message}</span>
      </motion.div>
    </AnimatePresence>
  )
}

// Empty state component
interface EmptyStateProps {
  icon?: React.ReactNode
  title: string
  description?: string
  action?: React.ReactNode
  className?: string
}

export function EmptyState({
  icon,
  title,
  description,
  action,
  className
}: EmptyStateProps) {
  return (
    <div className={cn(
      'flex flex-col items-center justify-center py-12 px-6 text-center',
      className
    )}>
      {icon && (
        <div className="w-16 h-16 rounded-full bg-dark-800 flex items-center justify-center mb-4 text-dark-400">
          {icon}
        </div>
      )}
      <h3 className="text-lg font-semibold text-white mb-2">{title}</h3>
      {description && (
        <p className="text-dark-400 max-w-sm mb-6">{description}</p>
      )}
      {action}
    </div>
  )
}
