import { useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { KEYBOARD_SHORTCUTS } from '@/utils/constants'

interface UseKeyboardNavigationOptions {
  enabled?: boolean
}

export function useKeyboardNavigation(options: UseKeyboardNavigationOptions = {}) {
  const { enabled = true } = options
  const navigate = useNavigate()

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    // Don't trigger shortcuts when typing in inputs
    if (
      e.target instanceof HTMLInputElement ||
      e.target instanceof HTMLTextAreaElement ||
      e.target instanceof HTMLSelectElement
    ) {
      return
    }

    // Check for modifier keys
    if (e.ctrlKey || e.metaKey) {
      return
    }

    switch (e.key.toLowerCase()) {
      case KEYBOARD_SHORTCUTS.HOME:
        e.preventDefault()
        navigate('/')
        break
      case KEYBOARD_SHORTCUTS.ANALYZE:
        e.preventDefault()
        navigate('/analyze')
        break
      case KEYBOARD_SHORTCUTS.MODELS:
        e.preventDefault()
        navigate('/models')
        break
      case KEYBOARD_SHORTCUTS.ABOUT:
        e.preventDefault()
        navigate('/about')
        break
      case '?':
        // Show keyboard shortcuts help
        e.preventDefault()
        // Could dispatch an event or show a modal
        break
    }
  }, [navigate])

  useEffect(() => {
    if (!enabled) return

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [enabled, handleKeyDown])
}

// Hook for focus management
export function useFocusReturn() {
  const previousFocusRef = { current: null as HTMLElement | null }

  const saveFocus = useCallback(() => {
    previousFocusRef.current = document.activeElement as HTMLElement
  }, [])

  const restoreFocus = useCallback(() => {
    previousFocusRef.current?.focus()
  }, [])

  return { saveFocus, restoreFocus }
}
