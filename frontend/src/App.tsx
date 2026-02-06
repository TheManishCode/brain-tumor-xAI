import { Routes, Route } from 'react-router-dom'
import { Layout } from './components/layout/Layout'
import { HomePage } from './pages/HomePage'
import { AnalyzePage } from './pages/AnalyzePage'
import { AboutPage } from './pages/AboutPage'
import { ModelsPage } from './pages/ModelsPage'
import { AccessibilityProvider } from './context/AccessibilityContext'
import { HealthProvider } from './context/HealthContext'
import { AnalysisProvider } from './context/AnalysisContext'
import { ChatbotV2 } from './components/features'

function App() {
  return (
    <AccessibilityProvider>
      <HealthProvider>
        <AnalysisProvider>
          <Layout>
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/analyze" element={<AnalyzePage />} />
              <Route path="/models" element={<ModelsPage />} />
              <Route path="/about" element={<AboutPage />} />
            </Routes>
          </Layout>
          {/* Global AI Chatbot V2 - Agentic RAG */}
          <ChatbotV2 />
        </AnalysisProvider>
      </HealthProvider>
    </AccessibilityProvider>
  )
}

export default App
