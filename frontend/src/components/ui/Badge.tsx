import { cn } from '@/utils/helpers'

interface BadgeProps {
  variant?: 'default' | 'primary' | 'success' | 'warning' | 'danger' | 'info'
  size?: 'sm' | 'md'
  children: React.ReactNode
  className?: string
  icon?: React.ReactNode
}

export function Badge({ 
  variant = 'default', 
  size = 'md',
  children, 
  className,
  icon 
}: BadgeProps) {
  const variants = {
    default: 'bg-dark-700/50 text-dark-300 border-dark-600',
    primary: 'bg-primary-500/10 text-primary-400 border-primary-500/20',
    success: 'bg-medical-green/10 text-medical-green border-medical-green/20',
    warning: 'bg-medical-yellow/10 text-medical-yellow border-medical-yellow/20',
    danger: 'bg-medical-red/10 text-medical-red border-medical-red/20',
    info: 'bg-medical-blue/10 text-medical-blue border-medical-blue/20'
  }

  const sizes = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-3 py-1 text-xs'
  }

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 font-medium rounded-full border',
        variants[variant],
        sizes[size],
        className
      )}
    >
      {icon && <span className="flex-shrink-0">{icon}</span>}
      {children}
    </span>
  )
}

// Pulsing status badge
export function StatusBadge({ 
  status, 
  label 
}: { 
  status: 'online' | 'offline' | 'loading'
  label?: string 
}) {
  const statusConfig = {
    online: {
      color: 'bg-medical-green',
      text: label || 'Online',
      textColor: 'text-medical-green'
    },
    offline: {
      color: 'bg-medical-red',
      text: label || 'Offline',
      textColor: 'text-medical-red'
    },
    loading: {
      color: 'bg-medical-yellow',
      text: label || 'Checking...',
      textColor: 'text-medical-yellow'
    }
  }

  const config = statusConfig[status]

  return (
    <div className="flex items-center gap-2">
      <span className="relative flex h-2.5 w-2.5">
        {status !== 'offline' && (
          <span
            className={cn(
              'animate-ping absolute inline-flex h-full w-full rounded-full opacity-75',
              config.color
            )}
          />
        )}
        <span
          className={cn(
            'relative inline-flex rounded-full h-2.5 w-2.5',
            config.color
          )}
        />
      </span>
      <span className={cn('text-sm font-medium', config.textColor)}>
        {config.text}
      </span>
    </div>
  )
}
