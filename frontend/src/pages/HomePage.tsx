import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  MagnifyingGlassIcon, 
  BeakerIcon,
  ArrowRightIcon,
} from '@heroicons/react/24/outline'
import { Button, Badge, Card } from '@/components/ui'
import { FeatureGrid, StatsRow, AccessibilityButton } from '@/components/features'
import { STATS, TUMOR_CLASSES } from '@/utils/constants'

export function HomePage() {
  return (
    <>
      {/* Hero Section */}
      <section className="relative min-h-[90vh] flex items-center overflow-hidden">
        {/* Background effects */}
        <div className="absolute inset-0 bg-grid-pattern opacity-30" />
        <div className="absolute inset-0 bg-gradient-to-b from-dark-950 via-dark-950/50 to-dark-950" />
        
        {/* Animated gradient orbs */}
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary-600/20 rounded-full blur-3xl animate-pulse-slow" />
        <div className="absolute bottom-1/4 right-1/4 w-64 h-64 bg-primary-400/10 rounded-full blur-3xl animate-pulse-slow delay-1000" />

        <div className="container-custom relative z-10">
          <div className="max-w-4xl mx-auto text-center">
            {/* Heading */}
            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold text-white mb-6 leading-tight"
            >
              Brain Tumor
              <br />
              <span className="gradient-text">Classification</span>
            </motion.h1>

            {/* Subtitle */}
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="text-lg md:text-xl text-dark-300 mb-8 max-w-2xl mx-auto text-balance"
            >
              Advanced deep learning ensemble for accurate brain MRI analysis. 
              Classify tumors into glioma, meningioma, pituitary, or healthy tissue 
              with state-of-the-art accuracy and explainability.
            </motion.p>

            {/* CTA Buttons */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.3 }}
              className="flex flex-wrap justify-center gap-4 mb-12"
            >
              <Link to="/analyze">
                <Button size="lg" leftIcon={<MagnifyingGlassIcon className="w-5 h-5" />}>
                  Start Analysis
                  <ArrowRightIcon className="w-4 h-4 ml-1" />
                </Button>
              </Link>
              <Link to="/models">
                <Button variant="secondary" size="lg" leftIcon={<BeakerIcon className="w-5 h-5" />}>
                  View Models
                </Button>
              </Link>
            </motion.div>

            {/* Stats */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.4 }}
            >
              <StatsRow stats={STATS} />
            </motion.div>
          </div>
        </div>

        {/* Scroll indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
          className="absolute bottom-8 left-1/2 -translate-x-1/2"
        >
          <motion.div
            animate={{ y: [0, 8, 0] }}
            transition={{ repeat: Infinity, duration: 1.5 }}
            className="w-6 h-10 rounded-full border-2 border-dark-600 flex justify-center pt-2"
          >
            <div className="w-1.5 h-3 bg-dark-500 rounded-full" />
          </motion.div>
        </motion.div>
      </section>

      {/* Features Section */}
      <section id="features" className="section bg-dark-900/30">
        <div className="container-custom">
          {/* Section header */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Core Capabilities
            </h2>
            <p className="text-dark-400 max-w-2xl mx-auto">
              Built with clinical deployment in mind, our system combines multiple 
              neural networks with advanced techniques for reliable predictions.
            </p>
          </motion.div>

          <FeatureGrid />
        </div>
      </section>

      {/* Classification Types Section */}
      <section className="section">
        <div className="container-custom">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Tumor Categories
            </h2>
            <p className="text-dark-400 max-w-2xl mx-auto">
              Our model can identify and classify brain tumors into four distinct categories
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {Object.values(TUMOR_CLASSES).map((tumor, index) => (
              <motion.div
                key={tumor.name}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
              >
                <Card variant="hover" className="h-full text-center">
                  <div 
                    className="w-16 h-16 rounded-2xl mx-auto mb-4 flex items-center justify-center text-3xl"
                    style={{ backgroundColor: `${tumor.color}15` }}
                  >
                    {tumor.icon}
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2">
                    {tumor.displayName}
                  </h3>
                  <p className="text-sm text-dark-400">
                    {tumor.description.substring(0, 100)}...
                  </p>
                  <Badge
                    variant={tumor.severity === 'normal' ? 'success' : tumor.severity === 'high' ? 'danger' : 'warning'}
                    className="mt-4"
                  >
                    {tumor.severity === 'normal' ? 'Normal' : tumor.severity === 'high' ? 'High Priority' : 'Moderate'}
                  </Badge>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="section">
        <div className="container-custom">
          <Card className="relative overflow-hidden bg-gradient-to-br from-primary-900/50 to-dark-900 border-primary-500/20">
            {/* Background decoration */}
            <div className="absolute top-0 right-0 w-96 h-96 bg-primary-500/10 rounded-full blur-3xl" />
            
            <div className="relative z-10 text-center py-12">
              <motion.h2
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                className="text-3xl md:text-4xl font-bold text-white mb-4"
              >
                Ready to Analyze?
              </motion.h2>
              <motion.p
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.1 }}
                className="text-dark-300 max-w-xl mx-auto mb-8"
              >
                Upload a brain MRI scan and get instant AI-powered analysis 
                with detailed predictions and explainability.
              </motion.p>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.2 }}
              >
                <Link to="/analyze">
                  <Button size="lg">
                    Start Free Analysis
                    <ArrowRightIcon className="w-5 h-5" />
                  </Button>
                </Link>
              </motion.div>
            </div>
          </Card>
        </div>
      </section>

      {/* Accessibility button */}
      <AccessibilityButton />
    </>
  )
}
