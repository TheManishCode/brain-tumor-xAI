import { useState, useCallback, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ArrowPathIcon } from '@heroicons/react/24/outline'
import { Button, Card, Alert, Toggle } from '@/components/ui'
import { AnalysisLoader } from '@/components/ui/Loading'
import { ImageUpload, ResultsDisplay, AccessibilityButton, ExplainabilityPanel } from '@/components/features'
import { useAnalysis } from '@/hooks'
import { useHealth } from '@/context/HealthContext'
import { useAnalysisContext } from '@/context/AnalysisContext'
import { analyzeWithExplainability } from '@/services/api'
import type { UploadedFile } from '@/types'

export function AnalyzePage() {
  const { isOnline, isLoading: healthLoading } = useHealth()
  const { setCurrentAnalysis, clearAnalysis } = useAnalysisContext()
  const [uploadedFile, setUploadedFile] = useState<UploadedFile | null>(null)
  const [options, setOptions] = useState({
    useTta: true,
    includeGradcam: true,
    advancedExplainability: false
  })
  const [advancedResults, setAdvancedResults] = useState<any>(null)
  const [advancedLoading, setAdvancedLoading] = useState(false)

  const { 
    status, 
    progress, 
    result, 
    error, 
    analyze, 
    reset,
    isAnalyzing 
  } = useAnalysis({
    useTta: options.useTta,
    includeGradcam: options.includeGradcam
  })

  // Update global analysis context when result changes
  useEffect(() => {
    if (result) {
      setCurrentAnalysis(result)
    }
  }, [result, setCurrentAnalysis])

  const handleFileSelect = useCallback((file: UploadedFile) => {
    setUploadedFile(file)
    reset()
  }, [reset])

  const handleClear = useCallback(() => {
    setUploadedFile(null)
    setAdvancedResults(null)
    clearAnalysis()
    reset()
  }, [reset, clearAnalysis])

  const handleAnalyze = async () => {
    if (!uploadedFile) return
    
    try {
      await analyze(uploadedFile.file)
      
      // If advanced explainability is enabled, fetch detailed analysis
      if (options.advancedExplainability) {
        setAdvancedLoading(true)
        try {
          const advancedData = await analyzeWithExplainability(uploadedFile.file)
          // Extract the explainability data from the response
          if (advancedData.explainability) {
            setAdvancedResults(advancedData.explainability)
          } else {
            setAdvancedResults(null)
          }
        } catch {
          setAdvancedResults(null)
        } finally {
          setAdvancedLoading(false)
        }
      }
    } catch (err) {
      // Error is handled by the hook
    }
  }

  const handleNewScan = () => {
    setUploadedFile(null)
    setAdvancedResults(null)
    clearAnalysis()
    reset()
  }

  return (
    <>
      <section className="section">
        <div className="container-custom">
          {/* Page header */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-12"
          >
            <h1 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Analyze Brain MRI
            </h1>
            <p className="text-dark-400 max-w-2xl mx-auto">
              Upload a brain MRI scan for instant AI-powered analysis. 
              Our ensemble of neural networks will classify the scan and provide 
              detailed predictions with explainability.
            </p>
          </motion.div>

          {/* Server status warning */}
          {!healthLoading && !isOnline && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-8"
            >
              <Alert variant="warning" title="Server Offline">
                The analysis server is currently offline. Please make sure the Flask backend 
                is running on port 5000 before uploading images.
              </Alert>
            </motion.div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Left panel: Upload */}
            <div className="lg:col-span-2">
              <Card padding="lg">
                <AnimatePresence mode="wait">
                  {status === 'complete' && result ? (
                    <motion.div
                      key="results"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                    >
                      {/* Results header with new scan button */}
                      <div className="flex items-center justify-between mb-6">
                        <h2 className="text-xl font-semibold text-white">Analysis Results</h2>
                        <Button 
                          variant="secondary" 
                          size="sm"
                          onClick={handleNewScan}
                          leftIcon={<ArrowPathIcon className="w-4 h-4" />}
                        >
                          New Scan
                        </Button>
                      </div>
                      
                      <ResultsDisplay result={result} showGradcam={options.includeGradcam} />
                      
                      {/* Advanced Explainability Panel */}
                      {options.advancedExplainability && (
                        <div className="mt-6">
                          {advancedLoading ? (
                            <div className="flex items-center justify-center py-8">
                              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500"></div>
                              <span className="ml-3 text-dark-400">Loading advanced explainability...</span>
                            </div>
                          ) : advancedResults ? (
                            <ExplainabilityPanel data={advancedResults} />
                          ) : null}
                        </div>
                      )}
                    </motion.div>
                  ) : isAnalyzing ? (
                    <motion.div
                      key="loading"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                    >
                      <AnalysisLoader progress={progress} />
                    </motion.div>
                  ) : (
                    <motion.div
                      key="upload"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                    >
                      <h2 className="text-xl font-semibold text-white mb-6">
                        Upload MRI Image
                      </h2>
                      
                      <ImageUpload
                        onFileSelect={handleFileSelect}
                        onClear={handleClear}
                        file={uploadedFile}
                        disabled={isAnalyzing}
                        error={error}
                      />

                      {/* Error display */}
                      {error && status === 'error' && (
                        <motion.div
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          className="mt-4"
                        >
                          <Alert variant="error" title="Analysis Failed">
                            {error}
                          </Alert>
                        </motion.div>
                      )}
                    </motion.div>
                  )}
                </AnimatePresence>
              </Card>
            </div>

            {/* Right panel: Options & Actions */}
            <div className="space-y-6">
              {/* Analysis Options */}
              <Card>
                <h3 className="text-lg font-semibold text-white mb-4">
                  Analysis Options
                </h3>
                
                <div className="space-y-4">
                  <Toggle
                    enabled={options.useTta}
                    onChange={(enabled) => setOptions(prev => ({ ...prev, useTta: enabled }))}
                    label="Test-Time Augmentation"
                    description="Apply 7 augmentations for more robust predictions"
                  />
                  
                  <Toggle
                    enabled={options.includeGradcam}
                    onChange={(enabled) => setOptions(prev => ({ ...prev, includeGradcam: enabled }))}
                    label="Grad-CAM Visualization"
                    description="Generate attention heatmaps for explainability"
                  />
                  
                  <Toggle
                    enabled={options.advancedExplainability}
                    onChange={(enabled) => setOptions(prev => ({ ...prev, advancedExplainability: enabled }))}
                    label="Advanced Explainability"
                    description="Integrated Gradients, Saliency Maps, Uncertainty Analysis"
                  />
                </div>
              </Card>

              {/* Analyze Button */}
              {!result && (
                <Button
                  className="w-full"
                  size="lg"
                  onClick={handleAnalyze}
                  disabled={!uploadedFile || isAnalyzing || !isOnline}
                  isLoading={isAnalyzing}
                >
                  {isAnalyzing ? 'Analyzing...' : 'Analyze Image'}
                </Button>
              )}

              {/* Info cards */}
              <Card className="bg-dark-800/30">
                <h3 className="text-sm font-medium text-white mb-3">
                  💡 Tips for Best Results
                </h3>
                <ul className="text-xs text-dark-400 space-y-2">
                  <li className="flex items-start gap-2">
                    <span className="text-primary-400">•</span>
                    Use T1 or T2-weighted MRI scans
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-primary-400">•</span>
                    Axial plane images work best
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-primary-400">•</span>
                    Higher resolution improves accuracy
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-primary-400">•</span>
                    Enable TTA for more reliable results
                  </li>
                </ul>
              </Card>

              <Card className="bg-dark-800/30 border-medical-yellow/20">
                <h3 className="text-sm font-medium text-medical-yellow mb-2">
                  ⚠️ Disclaimer
                </h3>
                <p className="text-xs text-dark-400">
                  This tool is for research and educational purposes only. 
                  It should not be used as a substitute for professional medical 
                  diagnosis. Always consult with qualified healthcare providers.
                </p>
              </Card>
            </div>
          </div>
        </div>
      </section>

      <AccessibilityButton />
    </>
  )
}
