import { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from 'react'
import { checkHealth } from '@/services/api'
import type { HealthStatus } from '@/types'

interface HealthContextType {
  status: HealthStatus | null
  isLoading: boolean
  isOnline: boolean
  error: string | null
  refresh: () => Promise<void>
}

const HealthContext = createContext<HealthContextType | undefined>(undefined)

const HEALTH_CHECK_INTERVAL = 30000 // 30 seconds

export function HealthProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<HealthStatus | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      setIsLoading(true)
      setError(null)
      const health = await checkHealth()
      setStatus(health)
    } catch (err) {
      setError('Unable to connect to server')
      setStatus(null)
    } finally {
      setIsLoading(false)
    }
  }, [])

  // Initial check and periodic refresh
  useEffect(() => {
    refresh()
    
    const interval = setInterval(refresh, HEALTH_CHECK_INTERVAL)
    
    return () => clearInterval(interval)
  }, [refresh])

  const isOnline = status?.status === 'healthy' && status?.models_loaded === true

  return (
    <HealthContext.Provider value={{ status, isLoading, isOnline, error, refresh }}>
      {children}
    </HealthContext.Provider>
  )
}

export function useHealth() {
  const context = useContext(HealthContext)
  if (context === undefined) {
    throw new Error('useHealth must be used within a HealthProvider')
  }
  return context
}
