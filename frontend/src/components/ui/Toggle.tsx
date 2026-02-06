import { Switch } from '@headlessui/react'
import { cn } from '@/utils/helpers'

interface ToggleProps {
  enabled: boolean
  onChange: (enabled: boolean) => void
  label: string
  description?: string
  size?: 'sm' | 'md'
  disabled?: boolean
}

export function Toggle({
  enabled,
  onChange,
  label,
  description,
  size = 'md',
  disabled = false
}: ToggleProps) {
  const sizes = {
    sm: {
      switch: 'h-5 w-9',
      dot: 'h-3.5 w-3.5',
      translate: 'translate-x-4'
    },
    md: {
      switch: 'h-6 w-11',
      dot: 'h-4 w-4',
      translate: 'translate-x-5'
    }
  }

  const sizeConfig = sizes[size]

  return (
    <Switch.Group>
      <div className="flex items-center justify-between">
        <div className="flex flex-col">
          <Switch.Label className="text-sm font-medium text-white cursor-pointer">
            {label}
          </Switch.Label>
          {description && (
            <Switch.Description className="text-xs text-dark-400 mt-0.5">
              {description}
            </Switch.Description>
          )}
        </div>
        <Switch
          checked={enabled}
          onChange={onChange}
          disabled={disabled}
          className={cn(
            'relative inline-flex flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent',
            'transition-colors duration-200 ease-in-out',
            'focus:outline-none focus-visible:ring-2 focus-visible:ring-primary-500 focus-visible:ring-offset-2 focus-visible:ring-offset-dark-950',
            sizeConfig.switch,
            enabled ? 'bg-primary-600' : 'bg-dark-700',
            disabled && 'opacity-50 cursor-not-allowed'
          )}
        >
          <span className="sr-only">{label}</span>
          <span
            className={cn(
              'pointer-events-none inline-block transform rounded-full bg-white shadow-lg ring-0',
              'transition duration-200 ease-in-out',
              sizeConfig.dot,
              enabled ? sizeConfig.translate : 'translate-x-0.5',
              'mt-0.5 ml-0.5'
            )}
          />
        </Switch>
      </div>
    </Switch.Group>
  )
}

// Simple checkbox
interface CheckboxProps {
  checked: boolean
  onChange: (checked: boolean) => void
  label: string
  description?: string
  disabled?: boolean
}

export function Checkbox({
  checked,
  onChange,
  label,
  description,
  disabled = false
}: CheckboxProps) {
  return (
    <label className={cn(
      'flex items-start gap-3 cursor-pointer',
      disabled && 'opacity-50 cursor-not-allowed'
    )}>
      <div className="relative flex items-center justify-center mt-0.5">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          disabled={disabled}
          className="sr-only peer"
        />
        <div className={cn(
          'w-5 h-5 border-2 rounded transition-all duration-200',
          'peer-focus-visible:ring-2 peer-focus-visible:ring-primary-500 peer-focus-visible:ring-offset-2 peer-focus-visible:ring-offset-dark-950',
          checked 
            ? 'bg-primary-600 border-primary-600' 
            : 'bg-dark-800 border-dark-600'
        )}>
          {checked && (
            <svg className="w-full h-full text-white" viewBox="0 0 16 16" fill="none">
              <path
                d="M4 8l3 3 5-6"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          )}
        </div>
      </div>
      <div>
        <span className="text-sm font-medium text-white">{label}</span>
        {description && (
          <p className="text-xs text-dark-400 mt-0.5">{description}</p>
        )}
      </div>
    </label>
  )
}
