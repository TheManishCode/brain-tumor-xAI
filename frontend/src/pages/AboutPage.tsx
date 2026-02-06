import { motion } from 'framer-motion'
import { 
  AcademicCapIcon,
  BeakerIcon,
  CodeBracketIcon,
  HeartIcon,
  EnvelopeIcon
} from '@heroicons/react/24/outline'
import { Card, Button } from '@/components/ui'
import { AccessibilityButton } from '@/components/features'

export function AboutPage() {
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
            <div className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-5 p-1">
              <img src="/brain-slug.svg" className="w-full h-full" alt="" />
            </div>
            <h1 className="text-3xl md:text-4xl font-bold text-white mb-4">
              About MidLens
            </h1>
            <p className="text-dark-400 max-w-2xl mx-auto">
              A deep learning system for brain tumor classification, 
              built with modern techniques and best practices for medical AI.
            </p>
          </motion.div>

          {/* Mission */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mb-16"
          >
            <Card className="bg-gradient-to-br from-primary-900/30 to-dark-900">
              <div className="text-center py-8">
                <h2 className="text-2xl font-bold text-white mb-4">Our Mission</h2>
                <p className="text-dark-300 max-w-3xl mx-auto text-lg leading-relaxed">
                  To demonstrate how modern deep learning techniques can be applied to 
                  medical imaging classification, providing accurate, explainable, and 
                  reliable solutions that could assist healthcare professionals 
                  in their diagnostic workflows.
                </p>
              </div>
            </Card>
          </motion.div>

          {/* Features grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0 }}
            >
              <Card className="h-full text-center">
                <div className="w-16 h-16 rounded-2xl bg-primary-500/10 flex items-center justify-center mx-auto mb-4">
                  <AcademicCapIcon className="w-8 h-8 text-primary-400" />
                </div>
                <h3 className="text-xl font-semibold text-white mb-2">
                  Research-Grade
                </h3>
                <p className="text-dark-400">
                  Built on peer-reviewed techniques including ensemble learning, 
                  test-time augmentation, and Grad-CAM visualization.
                </p>
              </Card>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.1 }}
            >
              <Card className="h-full text-center">
                <div className="w-16 h-16 rounded-2xl bg-green-500/10 flex items-center justify-center mx-auto mb-4">
                  <BeakerIcon className="w-8 h-8 text-green-400" />
                </div>
                <h3 className="text-xl font-semibold text-white mb-2">
                  Deployment Ready
                </h3>
                <p className="text-dark-400">
                  REST API with health checks, error handling, and modular 
                  architecture suitable for deployment.
                </p>
              </Card>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.2 }}
            >
              <Card className="h-full text-center">
                <div className="w-16 h-16 rounded-2xl bg-blue-500/10 flex items-center justify-center mx-auto mb-4">
                  <CodeBracketIcon className="w-8 h-8 text-blue-400" />
                </div>
                <h3 className="text-xl font-semibold text-white mb-2">
                  Open Source
                </h3>
                <p className="text-dark-400">
                  Fully open source under MIT license. Explore the code, 
                  contribute, or adapt it for your own projects.
                </p>
              </Card>
            </motion.div>
          </div>

          {/* Tech Stack */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mb-16"
          >
            <h2 className="text-2xl font-bold text-white text-center mb-8">
              Technology Stack
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
              {[
                { name: 'Python', color: '#3776AB' },
                { name: 'PyTorch', color: '#EE4C2C' },
                { name: 'Flask', color: '#000000' },
                { name: 'React', color: '#61DAFB' },
                { name: 'TypeScript', color: '#3178C6' },
                { name: 'Tailwind CSS', color: '#06B6D4' },
              ].map((tech, index) => (
                <motion.div
                  key={tech.name}
                  initial={{ opacity: 0, scale: 0.9 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.05 }}
                >
                  <Card className="text-center py-4 hover:border-primary-500/30 transition-colors">
                    <span className="font-medium text-white">{tech.name}</span>
                  </Card>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Dataset Info */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mb-16"
          >
            <Card>
              <h2 className="text-xl font-semibold text-white mb-4">
                📊 Dataset Information
              </h2>
              <p className="text-dark-300 mb-4">
                The models were trained on the Brain Tumor MRI Dataset, which contains 
                labeled MRI images across four categories: Glioma, Meningioma, Pituitary, 
                and No Tumor.
              </p>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
                <div className="text-center p-4 rounded-xl bg-dark-800/50">
                  <div className="text-2xl font-bold text-primary-400">7,023</div>
                  <div className="text-sm text-dark-400">Total Images</div>
                </div>
                <div className="text-center p-4 rounded-xl bg-dark-800/50">
                  <div className="text-2xl font-bold text-primary-400">4</div>
                  <div className="text-sm text-dark-400">Classes</div>
                </div>
                <div className="text-center p-4 rounded-xl bg-dark-800/50">
                  <div className="text-2xl font-bold text-primary-400">224×224</div>
                  <div className="text-sm text-dark-400">Input Size</div>
                </div>
                <div className="text-center p-4 rounded-xl bg-dark-800/50">
                  <div className="text-2xl font-bold text-primary-400">RGB</div>
                  <div className="text-sm text-dark-400">Color Mode</div>
                </div>
              </div>
            </Card>
          </motion.div>

          {/* Disclaimer */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mb-16"
          >
            <Card className="border-medical-yellow/20 bg-medical-yellow/5">
              <h2 className="text-xl font-semibold text-medical-yellow mb-4">
                ⚠️ Important Disclaimer
              </h2>
              <p className="text-dark-300">
                MidLens is designed for <strong>research and educational purposes only</strong>. 
                It is NOT intended for clinical use or medical diagnosis. The predictions made by 
                this system should never be used as a substitute for professional medical advice, 
                diagnosis, or treatment. Always seek the guidance of qualified healthcare providers 
                with any questions regarding medical conditions.
              </p>
            </Card>
          </motion.div>

          {/* Contact / Links */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <Card className="text-center">
              <h2 className="text-xl font-semibold text-white mb-4">
                Connect & Contribute
              </h2>
              <p className="text-dark-400 mb-6">
                Interested in the project? Check out the source code, report issues, 
                or contribute to make it better.
              </p>
              <div className="flex flex-wrap justify-center gap-4">
                <a 
                  href="https://github.com/TheManishCode/brain-tumor-xAI" 
                  target="_blank" 
                  rel="noopener noreferrer"
                >
                  <Button variant="secondary" leftIcon={<CodeBracketIcon className="w-5 h-5" />}>
                    View on GitHub
                  </Button>
                </a>
                <a href="mailto:manishchauhanvns@gmail.com">
                  <Button variant="ghost" leftIcon={<EnvelopeIcon className="w-5 h-5" />}>
                    Contact
                  </Button>
                </a>
              </div>
              
              <div className="mt-8 pt-8 border-t border-dark-700">
                <p className="text-sm text-dark-500 flex items-center justify-center gap-2">
                  Made with <HeartIcon className="w-4 h-4 text-medical-red" /> for the ML community
                </p>
              </div>
            </Card>
          </motion.div>
        </div>
      </section>

      <AccessibilityButton />
    </>
  )
}
