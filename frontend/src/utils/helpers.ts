import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

/** Merge Tailwind classes with conflict resolution. */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/** Format bytes into human-readable size. */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`
}

/** Format a 0-1 value as a percentage string. */
export function formatPercentage(value: number, decimals = 1): string {
  return `${(value * 100).toFixed(decimals)}%`
}

/** Get Tailwind background classes for a severity level. */
export function getSeverityBg(severity: 'normal' | 'moderate' | 'high'): string {
  switch (severity) {
    case 'normal':
      return 'bg-medical-green/10 border-medical-green/20'
    case 'moderate':
      return 'bg-medical-yellow/10 border-medical-yellow/20'
    case 'high':
      return 'bg-medical-red/10 border-medical-red/20'
    default:
      return 'bg-dark-800 border-dark-700'
  }
}

/** Validate an image file for upload (type + size). */
export function validateImageFile(file: File): { valid: boolean; error?: string } {
  const maxSize = 10 * 1024 * 1024 // 10 MB
  const allowedTypes = ['image/jpeg', 'image/png', 'image/jpg', 'image/webp']

  if (!allowedTypes.includes(file.type)) {
    return {
      valid: false,
      error: 'Invalid file type. Please upload a JPEG, PNG, or WebP image.',
    }
  }

  if (file.size > maxSize) {
    return {
      valid: false,
      error: `File size too large. Maximum size is ${formatFileSize(maxSize)}.`,
    }
  }

  return { valid: true }
}

/** Create a data-URL preview from a File. */
export function createImagePreview(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as string)
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}
