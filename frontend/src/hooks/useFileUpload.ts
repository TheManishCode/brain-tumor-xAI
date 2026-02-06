import { useState, useCallback, useEffect } from 'react'
import { validateImageFile, createImagePreview } from '@/utils/helpers'
import type { UploadedFile } from '@/types'

interface UseFileUploadOptions {
  maxSize?: number
  acceptedTypes?: string[]
  onFileSelect?: (file: UploadedFile) => void
  onError?: (error: string) => void
}

export function useFileUpload(options: UseFileUploadOptions = {}) {
  const {
    maxSize = 10 * 1024 * 1024, // 10MB
    acceptedTypes = ['image/jpeg', 'image/png', 'image/jpg', 'image/webp'],
    onFileSelect,
    onError
  } = options

  const [file, setFile] = useState<UploadedFile | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleFile = useCallback(async (rawFile: File) => {
    // Validate file
    const validation = validateImageFile(rawFile)
    if (!validation.valid) {
      setError(validation.error || 'Invalid file')
      onError?.(validation.error || 'Invalid file')
      return false
    }

    // Check size
    if (rawFile.size > maxSize) {
      const errorMsg = `File too large. Maximum size is ${Math.round(maxSize / 1024 / 1024)}MB`
      setError(errorMsg)
      onError?.(errorMsg)
      return false
    }

    // Check type
    if (!acceptedTypes.includes(rawFile.type)) {
      const errorMsg = 'Invalid file type. Please upload JPEG, PNG, or WebP'
      setError(errorMsg)
      onError?.(errorMsg)
      return false
    }

    try {
      const preview = await createImagePreview(rawFile)
      
      const uploadedFile: UploadedFile = {
        file: rawFile,
        preview,
        name: rawFile.name,
        size: rawFile.size,
        type: rawFile.type
      }

      setFile(uploadedFile)
      setError(null)
      onFileSelect?.(uploadedFile)
      
      return true
    } catch {
      const errorMsg = 'Failed to read file'
      setError(errorMsg)
      onError?.(errorMsg)
      return false
    }
  }, [maxSize, acceptedTypes, onFileSelect, onError])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    const droppedFile = e.dataTransfer.files[0]
    if (droppedFile) {
      handleFile(droppedFile)
    }
  }, [handleFile])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      handleFile(selectedFile)
    }
  }, [handleFile])

  const clearFile = useCallback(() => {
    setFile(null)
    setError(null)
  }, [])

  // Clean up preview URL on unmount
  useEffect(() => {
    return () => {
      if (file?.preview.startsWith('blob:')) {
        URL.revokeObjectURL(file.preview)
      }
    }
  }, [file])

  return {
    file,
    isDragging,
    error,
    handleFile,
    handleDrop,
    handleDragOver,
    handleDragLeave,
    handleInputChange,
    clearFile
  }
}
