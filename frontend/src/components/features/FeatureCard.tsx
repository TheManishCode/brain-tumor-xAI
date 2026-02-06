import { motion } from 'framer-motion'
import { 
  BoltIcon, 
  ArrowPathIcon, 
  ChartBarIcon, 
  EyeIcon,
  CpuChipIcon,
  ShieldCheckIcon,
  BeakerIcon,
  ClockIcon
} from '@heroicons/react/24/outline'
import { Card } from '@/components/ui'
import { cn } from '@/utils/helpers'

interface FeatureCardProps {
  title: string
  description: string
  icon: React.ReactNode
  color: 'blue' | 'purple' | 'green' | 'orange' | 'red' | 'indigo'
  delay?: number
}

const colorMap = {
  blue: {
    bg: 'bg-medical-blue/10',
    border: 'group-hover:border-medical-blue/30',
    icon: 'text-medical-blue'
  },
  purple: {
    bg: 'bg-purple-500/10',
    border: 'group-hover:border-purple-500/30',
    icon: 'text-purple-400'
  },
  green: {
    bg: 'bg-medical-green/10',
    border: 'group-hover:border-medical-green/30',
    icon: 'text-medical-green'
  },
  orange: {
    bg: 'bg-medical-orange/10',
    border: 'group-hover:border-medical-orange/30',
    icon: 'text-medical-orange'
  },
  red: {
    bg: 'bg-medical-red/10',
    border: 'group-hover:border-medical-red/30',
    icon: 'text-medical-red'
  },
  indigo: {
    bg: 'bg-primary-500/10',
    border: 'group-hover:border-primary-500/30',
    icon: 'text-primary-400'
  }
}

export function FeatureCard({ title, description, icon, color, delay = 0 }: FeatureCardProps) {
  const colors = colorMap[color]

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5, delay }}
    >
      <Card 
        variant="hover" 
        className={cn(
          'group h-full transition-all duration-300',
          colors.border
        )}
      >
        <div className={cn(
          'w-12 h-12 rounded-xl flex items-center justify-center mb-4',
          colors.bg
        )}>
          <div className={cn('w-6 h-6', colors.icon)}>
            {icon}
          </div>
        </div>
        <h3 className="text-lg font-semibold text-white mb-2">
          {title}
        </h3>
        <p className="text-sm text-dark-400 leading-relaxed">
          {description}
        </p>
      </Card>
    </motion.div>
  )
}

// Pre-defined feature list component
export function FeatureGrid() {
  const features = [
    {
      title: 'Weighted Ensemble',
      description: 'Three state-of-the-art CNNs combined with F1-optimized voting weights for superior accuracy.',
      icon: <BoltIcon />,
      color: 'blue' as const
    },
    {
      title: 'Test-Time Augmentation',
      description: '7 augmentations per model for robust predictions even on images from unknown sources.',
      icon: <ArrowPathIcon />,
      color: 'purple' as const
    },
    {
      title: 'Calibrated Confidence',
      description: 'Temperature-scaled probabilities for reliable uncertainty estimation in clinical settings.',
      icon: <ChartBarIcon />,
      color: 'green' as const
    },
    {
      title: 'Grad-CAM Explainability',
      description: 'Visual attention maps showing exactly what the model focuses on for each prediction.',
      icon: <EyeIcon />,
      color: 'orange' as const
    },
    {
      title: 'GPU Acceleration',
      description: 'Optimized inference with CUDA support for fast predictions on compatible hardware.',
      icon: <CpuChipIcon />,
      color: 'indigo' as const
    },
    {
      title: 'Privacy-First',
      description: 'All processing happens locally. Your medical images never leave your device.',
      icon: <ShieldCheckIcon />,
      color: 'green' as const
    },
    {
      title: 'CLAHE Preprocessing',
      description: 'Contrast-limited adaptive histogram equalization for enhanced feature extraction.',
      icon: <BeakerIcon />,
      color: 'blue' as const
    },
    {
      title: 'Real-Time Analysis',
      description: 'Get comprehensive results in under 2 seconds with our optimized pipeline.',
      icon: <ClockIcon />,
      color: 'purple' as const
    }
  ]

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {features.map((feature, index) => (
        <FeatureCard
          key={feature.title}
          {...feature}
          delay={index * 0.1}
        />
      ))}
    </div>
  )
}
