import { type ReactNode, useEffect } from 'react'
import { useLocation } from 'react-router-dom'
import { Header } from './Header'
import { Footer } from './Footer'
import { useKeyboardNavigation } from '@/hooks'
import { useAccessibility } from '@/context/AccessibilityContext'

interface LayoutProps {
  children: ReactNode
}

export function Layout({ children }: LayoutProps) {
  const location = useLocation()
  const { announceMessage } = useAccessibility()
  
  // Enable keyboard navigation shortcuts
  useKeyboardNavigation()

  // Announce page changes to screen readers
  useEffect(() => {
    const pageName = location.pathname === '/' 
      ? 'Home' 
      : location.pathname.replace('/', '').charAt(0).toUpperCase() + location.pathname.slice(2)
    
    announceMessage(`Navigated to ${pageName} page`)
    
    // Focus main content on page change
    const mainContent = document.getElementById('main-content')
    if (mainContent) {
      mainContent.focus()
    }
  }, [location.pathname, announceMessage])

  return (
    <div className="min-h-screen flex flex-col bg-dark-950">
      {/* Skip Links (WCAG 2.4.1) */}
      <a 
        href="#main-content" 
        className="skip-link"
      >
        Skip to main content
      </a>
      <a 
        href="#navigation" 
        className="skip-link"
      >
        Skip to navigation
      </a>

      <Header />
      
      <main 
        id="main-content" 
        className="flex-1 pt-16"
        role="main"
        tabIndex={-1}
        aria-label="Main content"
      >
        {children}
      </main>
      
      <Footer />
    </div>
  )
}
