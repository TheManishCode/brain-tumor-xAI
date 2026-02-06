import { motion } from 'framer-motion'
import { Card } from '@/components/ui'
import { ModelGrid, ModelComparisonTable, AccessibilityButton } from '@/components/features'
import { 
  BeakerIcon, 
  CpuChipIcon, 
  ArrowPathIcon,
  ChartBarIcon,
  AdjustmentsHorizontalIcon 
} from '@heroicons/react/24/outline'

export function ModelsPage() {
  return (
    <>
      <section className="section">
        <div className="container-custom">
          {/* Page header */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-16"
          >
            <h1 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Model Details
            </h1>
            <p className="text-dark-400 max-w-2xl mx-auto">
              MidLens uses a weighted ensemble of three state-of-the-art 
              convolutional neural networks, each trained on brain MRI data 
              for optimal classification performance.
            </p>
          </motion.div>

          {/* Model cards */}
          <ModelGrid />

          {/* Comparison table */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mt-16"
          >
            <Card>
              <h2 className="text-xl font-semibold text-white mb-6">
                Model Comparison
              </h2>
              <ModelComparisonTable />
            </Card>
          </motion.div>
        </div>
      </section>

      {/* Technical Details Section */}
      <section className="section bg-dark-900/30">
        <div className="container-custom">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              How It Works
            </h2>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {/* Ensemble Method */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0 }}
            >
              <Card className="h-full">
                <div className="w-12 h-12 rounded-xl bg-primary-500/10 flex items-center justify-center mb-4">
                  <BeakerIcon className="w-6 h-6 text-primary-400" />
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">
                  Weighted Ensemble
                </h3>
                <p className="text-sm text-dark-400 mb-4">
                  Model predictions are combined using F1-score optimized weights, 
                  giving higher influence to better-performing models.
                </p>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-dark-500">EfficientNet-B3</span>
                    <span className="text-primary-400 font-mono">40%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-dark-500">ResNet-50</span>
                    <span className="text-primary-400 font-mono">35%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-dark-500">DenseNet-121</span>
                    <span className="text-primary-400 font-mono">25%</span>
                  </div>
                </div>
              </Card>
            </motion.div>

            {/* TTA */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.1 }}
            >
              <Card className="h-full">
                <div className="w-12 h-12 rounded-xl bg-purple-500/10 flex items-center justify-center mb-4">
                  <ArrowPathIcon className="w-6 h-6 text-purple-400" />
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">
                  Test-Time Augmentation
                </h3>
                <p className="text-sm text-dark-400 mb-4">
                  Each model runs 7 augmentations per image, averaging predictions 
                  for more robust and reliable results.
                </p>
                <div className="text-sm text-dark-500 space-y-1">
                  <p>• Original image</p>
                  <p>• Horizontal flip</p>
                  <p>• Vertical flip</p>
                  <p>• Center crop 256→224</p>
                  <p>• Rotation 90°</p>
                  <p>• Rotation 180°</p>
                  <p>• Rotation 270°</p>
                </div>
              </Card>
            </motion.div>

            {/* Temperature Scaling */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.2 }}
            >
              <Card className="h-full">
                <div className="w-12 h-12 rounded-xl bg-green-500/10 flex items-center justify-center mb-4">
                  <AdjustmentsHorizontalIcon className="w-6 h-6 text-green-400" />
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">
                  Temperature Scaling
                </h3>
                <p className="text-sm text-dark-400 mb-4">
                  Per-model temperature scaling calibrates confidence scores 
                  for reliable uncertainty estimation.
                </p>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-dark-500">EfficientNet-B3</span>
                    <span className="text-green-400 font-mono">T=1.15</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-dark-500">ResNet-50</span>
                    <span className="text-green-400 font-mono">T=1.22</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-dark-500">DenseNet-121</span>
                    <span className="text-green-400 font-mono">T=1.28</span>
                  </div>
                </div>
              </Card>
            </motion.div>

            {/* Preprocessing */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.3 }}
            >
              <Card className="h-full">
                <div className="w-12 h-12 rounded-xl bg-blue-500/10 flex items-center justify-center mb-4">
                  <CpuChipIcon className="w-6 h-6 text-blue-400" />
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">
                  Image Preprocessing
                </h3>
                <p className="text-sm text-dark-400 mb-4">
                  CLAHE enhancement and ImageNet normalization prepare images 
                  for optimal feature extraction.
                </p>
                <div className="text-sm text-dark-500 space-y-1">
                  <p>• Resize to 224×224</p>
                  <p>• CLAHE contrast enhancement</p>
                  <p>• Grayscale to 3-channel</p>
                  <p>• ImageNet normalization</p>
                </div>
              </Card>
            </motion.div>

            {/* Grad-CAM */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.4 }}
            >
              <Card className="h-full">
                <div className="w-12 h-12 rounded-xl bg-orange-500/10 flex items-center justify-center mb-4">
                  <ChartBarIcon className="w-6 h-6 text-orange-400" />
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">
                  Grad-CAM Explainability
                </h3>
                <p className="text-sm text-dark-400 mb-4">
                  Gradient-weighted Class Activation Mapping visualizes which 
                  regions influenced the model's decision.
                </p>
                <div className="text-sm text-dark-500 space-y-1">
                  <p>• Computed from EfficientNet-B3</p>
                  <p>• Highlights tumor regions</p>
                  <p>• Heatmap + overlay views</p>
                  <p>• Interpretable results</p>
                </div>
              </Card>
            </motion.div>

            {/* Training */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.5 }}
            >
              <Card className="h-full">
                <div className="w-12 h-12 rounded-xl bg-red-500/10 flex items-center justify-center mb-4">
                  <BeakerIcon className="w-6 h-6 text-red-400" />
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">
                  Training Pipeline
                </h3>
                <p className="text-sm text-dark-400 mb-4">
                  Transfer learning from ImageNet with domain-specific 
                  augmentation and hyperparameter tuning.
                </p>
                <div className="text-sm text-dark-500 space-y-1">
                  <p>• Brain Tumor MRI Dataset</p>
                  <p>• 80/10/10 train/val/test split</p>
                  <p>• Cross-entropy + label smoothing</p>
                  <p>• AdamW optimizer</p>
                </div>
              </Card>
            </motion.div>
          </div>
        </div>
      </section>

      <AccessibilityButton />
    </>
  )
}
