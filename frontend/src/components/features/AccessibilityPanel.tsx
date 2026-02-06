/**
 * MidLens Accessibility Panel
 * ============================
 * WCAG 2.1 AA Compliant accessibility controls
 * Modern, intuitive UI with comprehensive options
 */

import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  XMarkIcon,
  ArrowPathIcon,
  EyeIcon,
  SpeakerWaveIcon,
  CursorArrowRaysIcon,
  DocumentTextIcon,
  ComputerDesktopIcon,
  SparklesIcon,
  CheckIcon,
  ChevronRightIcon
} from '@heroicons/react/24/outline'
import { Toggle, Button } from '@/components/ui'
import { useAccessibility } from '@/context/AccessibilityContext'
import { cn } from '@/utils/helpers'

interface AccessibilityPanelProps {
  isOpen: boolean
  onClose: () => void
}

// =============================================================================
// ACCESSIBILITY ICON
// =============================================================================
function AccessibilityIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <circle cx="12" cy="4" r="2" />
      <path d="M19 13v-2c-1.54.02-3.09-.75-4.07-1.83l-1.29-1.43c-.17-.19-.38-.34-.61-.45-.01 0-.01-.01-.02-.01H13c-.35-.2-.75-.3-1.19-.26C10.76 7.11 10 8.04 10 9.09V15c0 1.1.9 2 2 2h5v5h2v-5.5c0-1.1-.9-2-2-2h-3v-3.45c1.29 1.07 3.25 1.94 5 1.95zm-6.17 5c-.41 1.16-1.52 2-2.83 2-1.66 0-3-1.34-3-3 0-1.31.84-2.41 2-2.83V12.1c-2.28.46-4 2.48-4 4.9 0 2.76 2.24 5 5 5 2.42 0 4.44-1.72 4.9-4h-2.07z" />
    </svg>
  )
}

// =============================================================================
// SLIDER COMPONENT
// =============================================================================
interface SliderProps {
  label: string
  description: string
  value: number
  min: number
  max: number
  step: number
  unit: string
  onChange: (value: number) => void
  id: string
  icon?: React.ReactNode
}

function AccessibilitySlider({ label, description, value, min, max, step, unit, onChange, id, icon }: SliderProps) {
  const percentage = ((value - min) / (max - min)) * 100
  
  return (
    <div className="space-y-3">
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-2">
          {icon && <span className="text-primary-400">{icon}</span>}
          <label htmlFor={id} className="text-sm font-medium text-white">
            {label}
          </label>
        </div>
        <span className="text-sm text-primary-400 font-mono bg-primary-500/10 px-2 py-0.5 rounded-md" aria-live="polite">
          {value}{unit}
        </span>
      </div>
      <p id={`${id}-desc`} className="text-xs text-dark-400">
        {description}
      </p>
      <div className="relative">
        <input
          type="range"
          id={id}
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          aria-describedby={`${id}-desc`}
          aria-valuemin={min}
          aria-valuemax={max}
          aria-valuenow={value}
          aria-valuetext={`${value}${unit}`}
          className="w-full h-2 bg-dark-700 rounded-lg appearance-none cursor-pointer 
                     focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 focus:ring-offset-dark-900
                     [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-5 [&::-webkit-slider-thumb]:h-5 
                     [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-primary-500 [&::-webkit-slider-thumb]:cursor-pointer
                     [&::-webkit-slider-thumb]:shadow-lg [&::-webkit-slider-thumb]:shadow-primary-500/30
                     [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-primary-400
                     [&::-webkit-slider-thumb]:transition-all [&::-webkit-slider-thumb]:hover:scale-110
                     [&::-moz-range-thumb]:w-5 [&::-moz-range-thumb]:h-5 [&::-moz-range-thumb]:rounded-full 
                     [&::-moz-range-thumb]:bg-primary-500 [&::-moz-range-thumb]:border-2 [&::-moz-range-thumb]:border-primary-400"
          style={{
            background: `linear-gradient(to right, rgb(var(--primary-500)) 0%, rgb(var(--primary-500)) ${percentage}%, rgb(55, 65, 81) ${percentage}%, rgb(55, 65, 81) 100%)`
          }}
        />
      </div>
      <div className="flex justify-between text-xs text-dark-500">
        <span>{min}{unit}</span>
        <span>{max}{unit}</span>
      </div>
    </div>
  )
}

// =============================================================================
// TOGGLE CARD COMPONENT
// =============================================================================
interface ToggleCardProps {
  icon: React.ReactNode
  iconBg?: string
  label: string
  description: string
  wcagRef?: string
  enabled: boolean
  onChange: (enabled: boolean) => void
}

function ToggleCard({ icon, iconBg = 'bg-primary-500/10', label, description, wcagRef, enabled, onChange }: ToggleCardProps) {
  return (
    <motion.div 
      className={cn(
        "flex items-start gap-4 p-4 rounded-xl transition-all duration-200",
        "border",
        enabled 
          ? "bg-primary-500/5 border-primary-500/30" 
          : "bg-dark-800/30 border-dark-700/50 hover:bg-dark-800/50 hover:border-dark-600"
      )}
      whileHover={{ scale: 1.01 }}
      whileTap={{ scale: 0.99 }}
    >
      <div className={cn(
        "w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 transition-colors",
        enabled ? "bg-primary-500/20 text-primary-400" : `${iconBg} text-primary-400/70`
      )}>
        {icon}
      </div>
      <div className="flex-1 min-w-0">
        <Toggle
          enabled={enabled}
          onChange={onChange}
          label={label}
          description={wcagRef ? `${description} (${wcagRef})` : description}
        />
      </div>
    </motion.div>
  )
}

// =============================================================================
// SECTION COMPONENT
// =============================================================================
function Section({ title, icon, children, defaultOpen = true }: { title: string; icon: React.ReactNode; children: React.ReactNode; defaultOpen?: boolean }) {
  const [isOpen, setIsOpen] = useState(defaultOpen)
  
  return (
    <section className="space-y-3">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between text-left group"
        aria-expanded={isOpen}
      >
        <h3 className="text-sm font-semibold text-dark-300 uppercase tracking-wider flex items-center gap-2">
          <span className="text-primary-400/70">{icon}</span>
          {title}
        </h3>
        <ChevronRightIcon className={cn(
          "w-4 h-4 text-dark-500 transition-transform",
          isOpen && "rotate-90"
        )} />
      </button>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            {children}
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  )
}

// =============================================================================
// KEYBOARD SHORTCUT COMPONENT
// =============================================================================
function KeyboardShortcut({ keys, action }: { keys: string[]; action: string }) {
  return (
    <li className="flex justify-between items-center py-2">
      <span className="text-sm text-dark-300">{action}</span>
      <div className="flex items-center gap-1">
        {keys.map((key, i) => (
          <span key={i}>
            <kbd className="px-2 py-1 rounded-md bg-dark-700 text-dark-200 font-mono text-xs border border-dark-600 shadow-sm">
              {key}
            </kbd>
            {i < keys.length - 1 && <span className="text-dark-500 mx-1">+</span>}
          </span>
        ))}
      </div>
    </li>
  )
}

// =============================================================================
// MAIN PANEL
// =============================================================================
export function AccessibilityPanel({ isOpen, onClose }: AccessibilityPanelProps) {
  const { settings, updateSettings, resetSettings, announceMessage } = useAccessibility()
  const panelRef = useRef<HTMLDivElement>(null)
  const closeButtonRef = useRef<HTMLButtonElement>(null)

  // Focus trap
  useEffect(() => {
    if (!isOpen) return

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose()
      }

      if (e.key === 'Tab' && panelRef.current) {
        const focusableElements = panelRef.current.querySelectorAll(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        )
        const firstElement = focusableElements[0] as HTMLElement
        const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement

        if (e.shiftKey && document.activeElement === firstElement) {
          e.preventDefault()
          lastElement.focus()
        } else if (!e.shiftKey && document.activeElement === lastElement) {
          e.preventDefault()
          firstElement.focus()
        }
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    closeButtonRef.current?.focus()

    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, onClose])

  // Announce when panel opens
  useEffect(() => {
    if (isOpen) {
      announceMessage('Accessibility settings panel opened. Use Tab to navigate options.')
    }
  }, [isOpen, announceMessage])

  const handleToggleChange = (key: keyof typeof settings, value: boolean, announcement: string) => {
    updateSettings({ [key]: value })
    announceMessage(`${announcement} ${value ? 'enabled' : 'disabled'}`)
  }

  const handleReset = () => {
    resetSettings()
    announceMessage('Accessibility settings reset to defaults', 'assertive')
  }

  // Count active settings
  const activeCount = [
    settings.reducedMotion,
    settings.highContrast,
    settings.largeText,
    settings.screenReaderMode,
    settings.focusHighlight
  ].filter(Boolean).length

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50"
            aria-hidden="true"
          />

          {/* Panel */}
          <motion.div
            ref={panelRef}
            initial={{ opacity: 0, x: 320 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 320 }}
            transition={{ type: 'spring', damping: 28, stiffness: 300 }}
            role="dialog"
            aria-modal="true"
            aria-labelledby="accessibility-title"
            aria-describedby="accessibility-desc"
            className="fixed right-0 top-0 h-full w-full max-w-md bg-gradient-to-b from-dark-850 to-dark-900 
                       border-l border-dark-700/80 shadow-2xl z-50 overflow-hidden flex flex-col"
          >
            {/* Header */}
            <header className="sticky top-0 bg-dark-900/95 backdrop-blur-xl border-b border-dark-700/50 p-6 z-10">
              <div className="flex items-start justify-between gap-4">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center shadow-lg shadow-primary-600/30">
                    <AccessibilityIcon className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h2 id="accessibility-title" className="text-xl font-semibold text-white">
                      Accessibility
                    </h2>
                    <p id="accessibility-desc" className="text-sm text-dark-400">
                      {activeCount > 0 ? (
                        <span className="flex items-center gap-1">
                          <CheckIcon className="w-3.5 h-3.5 text-green-400" />
                          {activeCount} setting{activeCount > 1 ? 's' : ''} active
                        </span>
                      ) : (
                        'Customize your experience'
                      )}
                    </p>
                  </div>
                </div>
                <button
                  ref={closeButtonRef}
                  onClick={onClose}
                  className="p-2 rounded-xl text-dark-400 hover:text-white hover:bg-dark-700/70 
                             focus:outline-none focus:ring-2 focus:ring-primary-500 transition-colors"
                  aria-label="Close accessibility settings"
                >
                  <XMarkIcon className="w-6 h-6" />
                </button>
              </div>
            </header>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-6 space-y-8">
              {/* Visual Settings */}
              <Section title="Visual" icon={<EyeIcon className="w-4 h-4" />}>
                <div className="space-y-3">
                  <ToggleCard
                    icon={<SparklesIcon className="w-5 h-5" />}
                    label="Reduce Motion"
                    description="Minimize animations and transitions"
                    wcagRef="WCAG 2.3.3"
                    enabled={settings.reducedMotion}
                    onChange={(enabled) => handleToggleChange('reducedMotion', enabled, 'Reduced motion')}
                  />

                  <ToggleCard
                    icon={<EyeIcon className="w-5 h-5" />}
                    label="High Contrast"
                    description="Increase contrast for better visibility"
                    wcagRef="WCAG 1.4.3"
                    enabled={settings.highContrast}
                    onChange={(enabled) => handleToggleChange('highContrast', enabled, 'High contrast')}
                  />

                  <ToggleCard
                    icon={<DocumentTextIcon className="w-5 h-5" />}
                    label="Large Text"
                    description="Increase base font size to 125%"
                    wcagRef="WCAG 1.4.4"
                    enabled={settings.largeText}
                    onChange={(enabled) => handleToggleChange('largeText', enabled, 'Large text')}
                  />

                  <ToggleCard
                    icon={<CursorArrowRaysIcon className="w-5 h-5" />}
                    label="Enhanced Focus"
                    description="Larger, more visible focus indicators"
                    wcagRef="WCAG 2.4.7"
                    enabled={settings.focusHighlight}
                    onChange={(enabled) => handleToggleChange('focusHighlight', enabled, 'Enhanced focus')}
                  />
                </div>
              </Section>

              {/* Screen Reader */}
              <Section title="Screen Reader" icon={<SpeakerWaveIcon className="w-4 h-4" />}>
                <div className="space-y-3">
                  <ToggleCard
                    icon={<SpeakerWaveIcon className="w-5 h-5" />}
                    label="Screen Reader Optimized"
                    description="Enhanced ARIA labels and navigation"
                    wcagRef="WCAG 4.1.2"
                    enabled={settings.screenReaderMode}
                    onChange={(enabled) => handleToggleChange('screenReaderMode', enabled, 'Screen reader mode')}
                  />
                </div>
              </Section>

              {/* Text Spacing */}
              <Section title="Text Spacing" icon={<DocumentTextIcon className="w-4 h-4" />} defaultOpen={false}>
                <div className="space-y-6 p-4 rounded-xl bg-dark-800/30 border border-dark-700/50">
                  <AccessibilitySlider
                    id="font-size"
                    label="Font Size"
                    description="Adjust the base font size"
                    value={settings.fontSize}
                    min={100}
                    max={200}
                    step={25}
                    unit="%"
                    onChange={(value) => {
                      updateSettings({ fontSize: value })
                      announceMessage(`Font size set to ${value}%`)
                    }}
                  />

                  <AccessibilitySlider
                    id="line-height"
                    label="Line Height"
                    description="Adjust spacing between lines of text"
                    value={settings.lineHeight}
                    min={1.5}
                    max={3}
                    step={0.25}
                    unit=""
                    onChange={(value) => {
                      updateSettings({ lineHeight: value })
                      announceMessage(`Line height set to ${value}`)
                    }}
                  />

                  <AccessibilitySlider
                    id="letter-spacing"
                    label="Letter Spacing"
                    description="Adjust spacing between letters"
                    value={settings.letterSpacing}
                    min={0}
                    max={0.2}
                    step={0.02}
                    unit="em"
                    onChange={(value) => {
                      updateSettings({ letterSpacing: value })
                      announceMessage(`Letter spacing set to ${value} em`)
                    }}
                  />
                </div>
              </Section>

              {/* Keyboard Shortcuts */}
              <Section title="Keyboard Shortcuts" icon={<ComputerDesktopIcon className="w-4 h-4" />} defaultOpen={false}>
                <div className="p-4 rounded-xl bg-dark-800/30 border border-dark-700/50">
                  <ul className="divide-y divide-dark-700/50" role="list">
                    <KeyboardShortcut keys={['Alt', 'A']} action="Open this panel" />
                    <KeyboardShortcut keys={['Alt', 'C']} action="Toggle AI chat" />
                    <KeyboardShortcut keys={['Tab']} action="Navigate elements" />
                    <KeyboardShortcut keys={['Shift', 'Tab']} action="Navigate backwards" />
                    <KeyboardShortcut keys={['Escape']} action="Close dialogs" />
                    <KeyboardShortcut keys={['Enter']} action="Activate buttons" />
                    <KeyboardShortcut keys={['Space']} action="Toggle switches" />
                  </ul>
                </div>
              </Section>
            </div>

            {/* Footer */}
            <footer className="sticky bottom-0 p-6 bg-dark-900/95 backdrop-blur-xl border-t border-dark-700/50 space-y-4">
              {/* Reset Button */}
              <Button
                variant="secondary"
                onClick={handleReset}
                leftIcon={<ArrowPathIcon className="w-4 h-4" />}
                className="w-full"
              >
                Reset to Defaults
              </Button>

              {/* WCAG Compliance Info */}
              <div className="p-4 rounded-xl bg-gradient-to-r from-primary-500/5 to-emerald-500/5 border border-primary-500/20">
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 rounded-lg bg-primary-500/10 flex items-center justify-center flex-shrink-0">
                    <CheckIcon className="w-4 h-4 text-primary-400" />
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-primary-400">
                      WCAG 2.1 AA Compliant
                    </h4>
                    <p className="text-xs text-dark-400 mt-1">
                      This application follows Web Content Accessibility Guidelines for 
                      contrast ratios, keyboard navigation, and screen reader support.
                    </p>
                  </div>
                </div>
              </div>
            </footer>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}

// =============================================================================
// FLOATING BUTTON
// =============================================================================
export function AccessibilityButton() {
  const [isOpen, setIsOpen] = useState(false)
  const { settings, announceMessage } = useAccessibility()

  // Listen for keyboard shortcut event
  useEffect(() => {
    const handleOpenPanel = () => setIsOpen(true)
    window.addEventListener('accessibility:open-panel', handleOpenPanel)
    return () => window.removeEventListener('accessibility:open-panel', handleOpenPanel)
  }, [])

  // Count active settings for badge
  const activeCount = [
    settings.reducedMotion,
    settings.highContrast,
    settings.largeText,
    settings.screenReaderMode,
    settings.focusHighlight
  ].filter(Boolean).length

  const handleClick = () => {
    setIsOpen(true)
    announceMessage('Opening accessibility settings')
  }

  return (
    <>
      <motion.button
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={handleClick}
        className={cn(
          "fixed bottom-6 left-6 z-40",
          "w-12 h-12 rounded-full",
          "bg-dark-800/90 backdrop-blur-sm",
          "text-primary-400 shadow-lg",
          "border border-dark-700",
          "hover:bg-dark-700 hover:border-primary-500/50 hover:text-primary-300",
          "focus:outline-none focus:ring-4 focus:ring-primary-500/30 focus:ring-offset-2 focus:ring-offset-dark-950",
          "transition-all duration-200",
          "flex items-center justify-center"
        )}
        aria-label={`Accessibility settings (Alt+A)${activeCount > 0 ? `. ${activeCount} settings active` : ''}`}
        title="Accessibility Settings"
      >
        <AccessibilityIcon className="w-6 h-6" />
        
        {/* Active settings badge */}
        {activeCount > 0 && (
          <span className="absolute -top-1 -right-1 w-5 h-5 rounded-full bg-primary-500 text-white text-[10px] font-bold flex items-center justify-center border-2 border-dark-900">
            {activeCount}
          </span>
        )}
      </motion.button>

      <AccessibilityPanel isOpen={isOpen} onClose={() => setIsOpen(false)} />
    </>
  )
}
