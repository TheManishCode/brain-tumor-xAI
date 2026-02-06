import { motion } from 'framer-motion'
import { cn } from '@/utils/helpers'

interface StatCardProps {
  label: string
  value: string | number
  description?: string
  delay?: number
}

export function StatCard({ label, value, description, delay = 0 }: StatCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      whileInView={{ opacity: 1, scale: 1 }}
      viewport={{ once: true }}
      transition={{ duration: 0.3, delay }}
      className="text-center"
    >
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.5, delay: delay + 0.1 }}
        className="text-4xl md:text-5xl font-bold gradient-text mb-2"
      >
        {value}
      </motion.div>
      <div className="text-sm text-dark-300 font-medium">{label}</div>
      {description && (
        <div className="text-xs text-dark-500 mt-1">{description}</div>
      )}
    </motion.div>
  )
}

// Stats row component
interface StatsRowProps {
  stats: Array<{
    label: string
    value: string | number
    description?: string
  }>
  className?: string
}

export function StatsRow({ stats, className }: StatsRowProps) {
  return (
    <div className={cn(
      'flex flex-wrap justify-center items-center gap-8 md:gap-16',
      className
    )}>
      {stats.map((stat, index) => (
        <div key={stat.label} className="flex items-center">
          <StatCard {...stat} delay={index * 0.1} />
          {index < stats.length - 1 && (
            <div className="hidden md:block w-px h-12 bg-dark-700 ml-8 md:ml-16" />
          )}
        </div>
      ))}
    </div>
  )
}

// Animated counter
interface AnimatedCounterProps {
  value: number
  suffix?: string
  prefix?: string
  duration?: number
  className?: string
}

export function AnimatedCounter({ 
  value, 
  suffix = '', 
  prefix = '',
  duration = 2,
  className 
}: AnimatedCounterProps) {
  return (
    <motion.span
      className={cn('font-bold', className)}
      initial={{ opacity: 0 }}
      whileInView={{ opacity: 1 }}
      viewport={{ once: true }}
      transition={{ duration: duration * 0.5 }}
    >
      <motion.span
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
      >
        {prefix}
        <motion.span
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: duration * 0.15 }}
        >
          {value}
        </motion.span>
        {suffix}
      </motion.span>
    </motion.span>
  )
}
