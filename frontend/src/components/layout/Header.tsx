import { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Bars3Icon, XMarkIcon } from '@heroicons/react/24/outline'
import { StatusBadge } from '@/components/ui'
import { useHealth } from '@/context/HealthContext'
import { NAV_ITEMS } from '@/utils/constants'
import { cn } from '@/utils/helpers'

export function Header() {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)
  const { isOnline, isLoading } = useHealth()
  const location = useLocation()

  const getStatus = () => {
    if (isLoading) return 'loading'
    return isOnline ? 'online' : 'offline'
  }

  return (
    <header className="fixed top-0 left-0 right-0 z-40 bg-dark-950/80 backdrop-blur-xl border-b border-dark-800/50">
      <nav 
        className="container-custom"
        role="navigation" 
        aria-label="Main navigation"
      >
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link 
            to="/" 
            className="flex items-center gap-3 group"
            aria-label="MidLens Home"
          >
            <div className="relative">
              <div className="w-10 h-10 rounded-xl flex items-center justify-center group-hover:shadow-glow transition-shadow p-0.5">
                <img src="/brain-slug.svg" alt="" className="w-full h-full" />
              </div>
            </div>
            <span className="text-xl font-bold text-white">
              Mid<span className="gradient-text">Lens</span>
            </span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-1">
            {NAV_ITEMS.map((item) => {
              const isActive = location.pathname === item.href
              return (
                <Link
                  key={item.href}
                  to={item.href}
                  className={cn(
                    'relative px-4 py-2 text-sm font-medium rounded-lg transition-colors',
                    'focus:outline-none focus-visible:ring-2 focus-visible:ring-primary-500',
                    isActive 
                      ? 'text-white' 
                      : 'text-dark-400 hover:text-white hover:bg-dark-800'
                  )}
                  aria-current={isActive ? 'page' : undefined}
                >
                  {item.label}
                  {isActive && (
                    <motion.div
                      layoutId="nav-indicator"
                      className="absolute inset-0 bg-dark-800 rounded-lg -z-10"
                      transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
                    />
                  )}
                </Link>
              )
            })}
          </div>

          {/* Right side: Status + Mobile menu button */}
          <div className="flex items-center gap-4">
            <div className="hidden sm:block">
              <StatusBadge 
                status={getStatus()} 
                label={isLoading ? 'Checking...' : isOnline ? 'API Online' : 'API Offline'} 
              />
            </div>

            {/* Mobile menu button */}
            <button
              type="button"
              className="md:hidden p-2 rounded-lg text-dark-400 hover:text-white hover:bg-dark-800 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-primary-500"
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              aria-expanded={isMobileMenuOpen}
              aria-controls="mobile-menu"
              aria-label={isMobileMenuOpen ? 'Close menu' : 'Open menu'}
            >
              {isMobileMenuOpen ? (
                <XMarkIcon className="w-6 h-6" />
              ) : (
                <Bars3Icon className="w-6 h-6" />
              )}
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        <AnimatePresence>
          {isMobileMenuOpen && (
            <motion.div
              id="mobile-menu"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="md:hidden overflow-hidden"
            >
              <div className="py-4 space-y-1 border-t border-dark-800">
                {NAV_ITEMS.map((item) => {
                  const isActive = location.pathname === item.href
                  return (
                    <Link
                      key={item.href}
                      to={item.href}
                      className={cn(
                        'block px-4 py-3 text-base font-medium rounded-lg transition-colors',
                        isActive
                          ? 'bg-dark-800 text-white'
                          : 'text-dark-400 hover:text-white hover:bg-dark-800'
                      )}
                      onClick={() => setIsMobileMenuOpen(false)}
                      aria-current={isActive ? 'page' : undefined}
                    >
                      {item.label}
                    </Link>
                  )
                })}
                <div className="pt-4 px-4 sm:hidden">
                  <StatusBadge 
                    status={getStatus()} 
                    label={isLoading ? 'Checking...' : isOnline ? 'API Online' : 'API Offline'} 
                  />
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </nav>
    </header>
  )
}
