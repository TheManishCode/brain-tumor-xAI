import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  CloudArrowUpIcon, 
  PhotoIcon,
  XMarkIcon,
  DocumentArrowUpIcon
} from '@heroicons/react/24/outline'
import { Button } from '@/components/ui'
import { useFileUpload } from '@/hooks'
import { cn, formatFileSize } from '@/utils/helpers'
import type { UploadedFile } from '@/types'

interface ImageUploadProps {
  onFileSelect: (file: UploadedFile) => void
  onClear: () => void
  file: UploadedFile | null
  disabled?: boolean
  error?: string | null
}

export function ImageUpload({ 
  onFileSelect, 
  onClear, 
  file, 
  disabled = false,
  error: externalError 
}: ImageUploadProps) {
  const {
    file: uploadedFile,
    isDragging,
    error: uploadError,
    handleFile,
    clearFile
  } = useFileUpload({
    onFileSelect,
    onError: () => {}
  })

  const error = externalError || uploadError

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0 && !disabled) {
      handleFile(acceptedFiles[0])
    }
  }, [handleFile, disabled])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/png': ['.png'],
      'image/webp': ['.webp']
    },
    maxSize: 10 * 1024 * 1024, // 10MB
    multiple: false,
    disabled,
    noClick: false,
    noKeyboard: false
  })

  const handleClear = () => {
    clearFile()
    onClear()
  }

  const displayFile = file || uploadedFile

  return (
    <div className="w-full">
      <AnimatePresence mode="wait">
        {!displayFile ? (
          <motion.div
            key="dropzone"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.2 }}
          >
            <div
              {...getRootProps()}
              className={cn(
                'relative flex flex-col items-center justify-center',
                'w-full min-h-[300px] p-8',
                'border-2 border-dashed rounded-2xl',
                'transition-all duration-300 cursor-pointer',
                'focus:outline-none focus-visible:ring-2 focus-visible:ring-primary-500',
                isDragActive || isDragging
                  ? 'border-primary-500 bg-primary-500/10'
                  : 'border-dark-600 hover:border-primary-500/50 hover:bg-dark-800/50',
                disabled && 'opacity-50 cursor-not-allowed',
                error && 'border-medical-red/50 bg-medical-red/5'
              )}
              role="button"
              aria-label="Upload brain MRI image. Click or drag and drop."
              tabIndex={0}
            >
              <input {...getInputProps()} />
              
              {/* Upload icon with animation */}
              <motion.div
                animate={{ 
                  y: isDragActive ? -10 : 0,
                  scale: isDragActive ? 1.1 : 1 
                }}
                className={cn(
                  'w-20 h-20 rounded-2xl flex items-center justify-center mb-6',
                  'bg-gradient-to-br from-primary-600/20 to-primary-400/20',
                  isDragActive && 'from-primary-600/30 to-primary-400/30'
                )}
              >
                {isDragActive ? (
                  <DocumentArrowUpIcon className="w-10 h-10 text-primary-400" />
                ) : (
                  <CloudArrowUpIcon className="w-10 h-10 text-primary-400" />
                )}
              </motion.div>

              {/* Text content */}
              <div className="text-center">
                <p className="text-lg font-medium text-white mb-2">
                  {isDragActive 
                    ? 'Drop your image here' 
                    : 'Drag & drop a brain MRI image'
                  }
                </p>
                <p className="text-sm text-dark-400 mb-4">
                  or click to browse your files
                </p>
                <p className="text-xs text-dark-500">
                  Supports: JPEG, PNG, WebP • Max size: 10MB
                </p>
              </div>

              {/* Sample images hint */}
              <div className="mt-6 flex items-center gap-2 text-xs text-dark-500">
                <PhotoIcon className="w-4 h-4" />
                <span>Use T1-weighted or T2-weighted MRI scans for best results</span>
              </div>
            </div>

            {/* Error message */}
            {error && (
              <motion.p
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-3 text-sm text-medical-red flex items-center gap-2"
                role="alert"
              >
                <XMarkIcon className="w-4 h-4" />
                {error}
              </motion.p>
            )}
          </motion.div>
        ) : (
          <motion.div
            key="preview"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.2 }}
            className="relative"
          >
            {/* Image preview */}
            <div className="relative rounded-2xl overflow-hidden bg-dark-800 border border-dark-700">
              <img
                src={displayFile.preview}
                alt={`Preview of ${displayFile.name}`}
                className="w-full h-auto max-h-[400px] object-contain mx-auto"
              />
              
              {/* Overlay with file info */}
              <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-dark-950/90 to-transparent p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-dark-700 flex items-center justify-center">
                      <PhotoIcon className="w-5 h-5 text-dark-300" />
                    </div>
                    <div>
                      <p className="text-sm font-medium text-white truncate max-w-[200px]">
                        {displayFile.name}
                      </p>
                      <p className="text-xs text-dark-400">
                        {formatFileSize(displayFile.size)}
                      </p>
                    </div>
                  </div>
                  
                  {/* Remove button */}
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation()
                      handleClear()
                    }}
                    disabled={disabled}
                    aria-label="Remove uploaded image"
                  >
                    <XMarkIcon className="w-5 h-5" />
                    Remove
                  </Button>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
