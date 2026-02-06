import { Link } from 'react-router-dom'
import { HeartIcon } from '@heroicons/react/24/solid'

export function Footer() {
  const currentYear = new Date().getFullYear()

  return (
    <footer 
      className="bg-dark-900/50 border-t border-dark-800"
      role="contentinfo"
    >
      <div className="container-custom py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
          {/* Brand */}
          <div className="md:col-span-2">
            <Link to="/" className="inline-flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-xl flex items-center justify-center p-0.5">
                <img src="/brain-slug.svg" alt="" className="w-full h-full" />
              </div>
              <span className="text-xl font-bold text-white">MidLens</span>
            </Link>
            <p className="text-dark-400 text-sm max-w-md mb-4">
              AI-powered brain tumor classification using a weighted ensemble of state-of-the-art 
              neural networks. Accurate, explainable, and clinically informed.
            </p>
            <p className="text-xs text-dark-500">
              ⚠️ This is a research tool and should not be used for clinical diagnosis.
            </p>
          </div>

          {/* Quick Links */}
          <div>
            <h3 className="text-sm font-semibold text-white mb-4">Navigation</h3>
            <ul className="space-y-2">
              <li>
                <Link to="/" className="text-sm text-dark-400 hover:text-white transition-colors">
                  Home
                </Link>
              </li>
              <li>
                <Link to="/analyze" className="text-sm text-dark-400 hover:text-white transition-colors">
                  Analyze Image
                </Link>
              </li>
              <li>
                <Link to="/models" className="text-sm text-dark-400 hover:text-white transition-colors">
                  Model Details
                </Link>
              </li>
              <li>
                <Link to="/about" className="text-sm text-dark-400 hover:text-white transition-colors">
                  About Project
                </Link>
              </li>
            </ul>
          </div>

          {/* Technical */}
          <div>
            <h3 className="text-sm font-semibold text-white mb-4">Technical</h3>
            <ul className="space-y-2">
              <li>
                <a 
                  href="https://github.com/TheManishCode/brain-tumor-xAI" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-sm text-dark-400 hover:text-white transition-colors"
                >
                  GitHub Repository
                </a>
              </li>
              <li>
                <Link to="/models" className="text-sm text-dark-400 hover:text-white transition-colors">
                  Model Architecture
                </Link>
              </li>
              <li>
                <a 
                  href="https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-sm text-dark-400 hover:text-white transition-colors"
                >
                  Training Dataset
                </a>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="pt-8 border-t border-dark-800 flex flex-col sm:flex-row justify-between items-center gap-4">
          <p className="text-sm text-dark-500">
            © {currentYear} MidLens. MIT License.
          </p>
          <p className="text-sm text-dark-500 flex items-center gap-1">
            Built with <HeartIcon className="w-4 h-4 text-medical-red" /> using React, PyTorch & Flask
          </p>
        </div>
      </div>
    </footer>
  )
}
