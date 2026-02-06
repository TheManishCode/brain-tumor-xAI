/**
 * Accessibility Context
 * ======================
 * WCAG 2.1 AA Compliant accessibility management
 */

import { 
  createContext, 
  useContext, 
  useState, 
  useEffect, 
  useCallback,
  type ReactNode 
} from 'react'
import type { AccessibilitySettings } from '@/types'

// =============================================================================
// TYPES
// =============================================================================

interface AccessibilityContextType {
  settings: AccessibilitySettings
  updateSettings: (settings: Partial<AccessibilitySettings>) => void
  resetSettings: () => void
  announceMessage: (message: string, priority?: 'polite' | 'assertive') => void
}

// =============================================================================
// DEFAULTS & HELPERS
// =============================================================================

const STORAGE_KEY = 'midlens-accessibility-v2'

const prefersReducedMotion = (): boolean => {
  if (typeof window === 'undefined') return false
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches
}

const prefersHighContrast = (): boolean => {
  if (typeof window === 'undefined') return false
  return window.matchMedia('(prefers-contrast: high)').matches
}

const getDefaultSettings = (): AccessibilitySettings => ({
  reducedMotion: prefersReducedMotion(),
  highContrast: prefersHighContrast(),
  largeText: false,
  screenReaderMode: false,
  focusHighlight: true,
  fontSize: 100,
  lineHeight: 1.5,
  letterSpacing: 0
})

// =============================================================================
// LIVE REGION MANAGER
// =============================================================================

let politeRegion: HTMLDivElement | null = null
let assertiveRegion: HTMLDivElement | null = null

function setupLiveRegions() {
  if (typeof document === 'undefined') return

  if (!politeRegion) {
    politeRegion = document.createElement('div')
    politeRegion.id = 'aria-live-polite'
    politeRegion.setAttribute('role', 'status')
    politeRegion.setAttribute('aria-live', 'polite')
    politeRegion.setAttribute('aria-atomic', 'true')
    politeRegion.className = 'sr-only'
    politeRegion.style.cssText = `
      position: absolute !important;
      width: 1px !important;
      height: 1px !important;
      padding: 0 !important;
      margin: -1px !important;
      overflow: hidden !important;
      clip: rect(0, 0, 0, 0) !important;
      white-space: nowrap !important;
      border: 0 !important;
    `
    document.body.appendChild(politeRegion)
  }

  if (!assertiveRegion) {
    assertiveRegion = document.createElement('div')
    assertiveRegion.id = 'aria-live-assertive'
    assertiveRegion.setAttribute('role', 'alert')
    assertiveRegion.setAttribute('aria-live', 'assertive')
    assertiveRegion.setAttribute('aria-atomic', 'true')
    assertiveRegion.className = 'sr-only'
    assertiveRegion.style.cssText = `
      position: absolute !important;
      width: 1px !important;
      height: 1px !important;
      padding: 0 !important;
      margin: -1px !important;
      overflow: hidden !important;
      clip: rect(0, 0, 0, 0) !important;
      white-space: nowrap !important;
      border: 0 !important;
    `
    document.body.appendChild(assertiveRegion)
  }
}

// =============================================================================
// CONTEXT
// =============================================================================

const AccessibilityContext = createContext<AccessibilityContextType | undefined>(undefined)

// =============================================================================
// PROVIDER
// =============================================================================

export function AccessibilityProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<AccessibilitySettings>(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem(STORAGE_KEY)
      if (saved) {
        try {
          return { ...getDefaultSettings(), ...JSON.parse(saved) }
        } catch {
          return getDefaultSettings()
        }
      }
    }
    return getDefaultSettings()
  })

  // Setup live regions on mount
  useEffect(() => {
    setupLiveRegions()
  }, [])

  // Apply settings to DOM
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings))
    
    const root = document.documentElement
    const body = document.body
    
    // --- Reduced Motion (WCAG 2.3.3) ---
    root.classList.toggle('reduce-motion', settings.reducedMotion)
    if (settings.reducedMotion) {
      root.style.setProperty('--motion-duration', '0.01ms')
    } else {
      root.style.removeProperty('--motion-duration')
    }
    
    // --- High Contrast (WCAG 1.4.3, 1.4.6) ---
    root.classList.toggle('high-contrast', settings.highContrast)
    
    // --- Font Size (WCAG 1.4.4 - Resize Text up to 200%) ---
    const fontSize = settings.largeText ? 125 : settings.fontSize
    root.classList.toggle('large-text', settings.largeText || settings.fontSize > 100)
    root.style.fontSize = `${fontSize}%`
    
    // --- Line Height (WCAG 1.4.12 - Text Spacing) ---
    body.style.lineHeight = `${settings.lineHeight}`
    
    // --- Letter Spacing (WCAG 1.4.12 - Text Spacing) ---
    body.style.letterSpacing = settings.letterSpacing > 0 ? `${settings.letterSpacing}em` : ''
    
    // --- Enhanced Focus (WCAG 2.4.7 - Focus Visible) ---
    root.classList.toggle('enhanced-focus', settings.focusHighlight)
    
    // --- Screen Reader Optimizations ---
    if (settings.screenReaderMode) {
      root.setAttribute('data-sr-mode', 'true')
    } else {
      root.removeAttribute('data-sr-mode')
    }
    
  }, [settings])

  // Listen for system preference changes
  useEffect(() => {
    const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
    const contrastQuery = window.matchMedia('(prefers-contrast: high)')

    const handleMotionChange = (e: MediaQueryListEvent) => {
      setSettings(prev => ({ ...prev, reducedMotion: e.matches }))
    }

    const handleContrastChange = (e: MediaQueryListEvent) => {
      setSettings(prev => ({ ...prev, highContrast: e.matches }))
    }

    motionQuery.addEventListener('change', handleMotionChange)
    contrastQuery.addEventListener('change', handleContrastChange)

    return () => {
      motionQuery.removeEventListener('change', handleMotionChange)
      contrastQuery.removeEventListener('change', handleContrastChange)
    }
  }, [])

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return

      // Alt + A = Open accessibility panel
      if (e.altKey && e.key.toLowerCase() === 'a') {
        e.preventDefault()
        window.dispatchEvent(new CustomEvent('accessibility:open-panel'))
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  const updateSettings = useCallback((newSettings: Partial<AccessibilitySettings>) => {
    setSettings(prev => ({ ...prev, ...newSettings }))
  }, [])

  const resetSettings = useCallback(() => {
    setSettings(getDefaultSettings())
    localStorage.removeItem(STORAGE_KEY)
  }, [])

  const announceMessage = useCallback((
    message: string, 
    priority: 'polite' | 'assertive' = 'polite'
  ) => {
    const region = priority === 'assertive' ? assertiveRegion : politeRegion
    if (!region) {
      setupLiveRegions()
      return
    }
    
    // Clear and set new message
    region.textContent = ''
    requestAnimationFrame(() => {
      region.textContent = message
      setTimeout(() => {
        region.textContent = ''
      }, 1000)
    })
  }, [])

  return (
    <AccessibilityContext.Provider 
      value={{ settings, updateSettings, resetSettings, announceMessage }}
    >
      {children}
    </AccessibilityContext.Provider>
  )
}

// =============================================================================
// HOOK
// =============================================================================

export function useAccessibility() {
  const context = useContext(AccessibilityContext)
  if (context === undefined) {
    throw new Error('useAccessibility must be used within an AccessibilityProvider')
  }
  return context
}
