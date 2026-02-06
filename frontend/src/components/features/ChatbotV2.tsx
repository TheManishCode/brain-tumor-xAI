/**
 * MidLens AI Chatbot V2
 * ======================
 * Redesigned chat interface with source verification
 */

import { useState, useRef, useEffect, useCallback, memo, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  ChatBubbleLeftRightIcon,
  PaperAirplaneIcon,
  XMarkIcon,
  SparklesIcon,
  ArrowPathIcon,
  LightBulbIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  MinusIcon,
  ArrowsPointingOutIcon,
  CheckBadgeIcon,
  DocumentTextIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline'
import { cn } from '@/utils/helpers'
import { useAnalysisContext } from '@/context/AnalysisContext'

// ─── Types ──────────────────────────────────────────
interface Source {
  title: string
  source_type: string
  reliability: number
}

interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: number
  sources?: Source[]
  confidence?: number
  model?: string
  error?: boolean
}

interface ChatState {
  messages: ChatMessage[]
  isLoading: boolean
  isOpen: boolean
  isMinimized: boolean
  isExpanded: boolean
}

// ─── Chat Logic Hook ────────────────────────────────
function useChatbot() {
  const { currentAnalysis } = useAnalysisContext()
  const [state, setState] = useState<ChatState>({
    messages: [],
    isLoading: false,
    isOpen: false,
    isMinimized: false,
    isExpanded: false,
  })

  const sessionId = useRef(`chat-${Date.now()}-${Math.random().toString(36).slice(2)}`)

  const suggestions = useMemo(() => {
    if (currentAnalysis?.prediction) {
      const name = currentAnalysis.prediction.display_name || currentAnalysis.prediction.class
      return [
        `What is ${name}?`,
        'Explain my analysis results',
        'What are the treatment options?',
        'What should I discuss with my doctor?',
      ]
    }
    return [
      'What types of brain tumors can you detect?',
      'How accurate is the AI analysis?',
      'What are common symptoms of brain tumors?',
    ]
  }, [currentAnalysis])

  const sendMessage = useCallback(
    async (text: string) => {
      if (!text.trim() || state.isLoading) return

      const userMsg: ChatMessage = {
        id: `u-${Date.now()}`,
        role: 'user',
        content: text.trim(),
        timestamp: Date.now(),
      }

      setState(s => ({ ...s, messages: [...s.messages, userMsg], isLoading: true }))

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: text,
            session_id: sessionId.current,
            analysis: currentAnalysis,
          }),
        })

        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data = await res.json()
        if (data.error) throw new Error(data.error)

        const assistantMsg: ChatMessage = {
          id: `a-${Date.now()}`,
          role: 'assistant',
          content: data.message || 'No response received.',
          timestamp: Date.now(),
          sources: data.sources,
          confidence: data.confidence,
          model: data.model_used,
        }

        setState(s => ({ ...s, messages: [...s.messages, assistantMsg], isLoading: false }))
      } catch {
        const errorMsg: ChatMessage = {
          id: `e-${Date.now()}`,
          role: 'assistant',
          content: 'Sorry, I encountered an error. Please try again.',
          timestamp: Date.now(),
          error: true,
        }
        setState(s => ({ ...s, messages: [...s.messages, errorMsg], isLoading: false }))
      }
    },
    [state.isLoading, currentAnalysis]
  )

  const clearChat = useCallback(async () => {
    try {
      await fetch('/api/chat/clear', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId.current }),
      })
    } catch {
      /* ignore */
    }
    setState(s => ({ ...s, messages: [] }))
  }, [])

  const toggle = useCallback(
    (key: keyof Pick<ChatState, 'isOpen' | 'isMinimized' | 'isExpanded'>) => {
      setState(s => ({ ...s, [key]: !s[key] }))
    },
    []
  )

  const open = useCallback(() => {
    setState(s => ({ ...s, isOpen: true, isMinimized: false }))
  }, [])

  return { ...state, suggestions, currentAnalysis, sendMessage, clearChat, toggle, open }
}

// ─── Main Component ─────────────────────────────────
export function ChatbotV2({ className }: { className?: string }) {
  const chat = useChatbot()
  const [input, setInput] = useState('')
  const messagesRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    messagesRef.current?.scrollTo({ top: messagesRef.current.scrollHeight, behavior: 'smooth' })
  }, [chat.messages, chat.isLoading])

  useEffect(() => {
    if (chat.isOpen && !chat.isMinimized) inputRef.current?.focus()
  }, [chat.isOpen, chat.isMinimized])

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.altKey && e.key === 'c') {
        e.preventDefault()
        chat.toggle('isOpen')
      }
      if (e.key === 'Escape' && chat.isOpen) chat.toggle('isOpen')
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [chat])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (input.trim()) {
      chat.sendMessage(input)
      setInput('')
      if (inputRef.current) inputRef.current.style.height = 'auto'
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  const handleTextareaResize = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)
    e.target.style.height = 'auto'
    e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px'
  }

  return (
    <>
      {/* ── Floating Toggle Button ── */}
      <AnimatePresence>
        {!chat.isOpen && (
          <motion.button
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0, opacity: 0 }}
            whileHover={{ scale: 1.08 }}
            whileTap={{ scale: 0.92 }}
            onClick={chat.open}
            className={cn(
              'fixed bottom-6 right-6 z-50 w-14 h-14 rounded-2xl',
              'bg-dark-800 border border-dark-600/60 text-dark-200',
              'shadow-xl shadow-black/30',
              'flex items-center justify-center transition-colors duration-200',
              'hover:border-primary-500/50 hover:text-primary-400',
              'focus:outline-none focus:ring-2 focus:ring-primary-500/50',
              className
            )}
            aria-label="Open AI Assistant (Alt+C)"
          >
            {chat.currentAnalysis && (
              <span className="absolute -top-1 -right-1 w-3.5 h-3.5 bg-emerald-500 rounded-full border-2 border-dark-900 animate-pulse" />
            )}
            <ChatBubbleLeftRightIcon className="w-6 h-6" />
          </motion.button>
        )}
      </AnimatePresence>

      {/* ── Chat Window ── */}
      <AnimatePresence>
        {chat.isOpen && !chat.isMinimized && (
          <motion.div
            initial={{ opacity: 0, y: 24, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 24, scale: 0.96 }}
            transition={{ type: 'spring', stiffness: 400, damping: 30 }}
            className={cn(
              'fixed bottom-6 right-6 z-50 flex flex-col',
              'bg-dark-900 border border-dark-700/60',
              'rounded-2xl shadow-2xl shadow-black/40 overflow-hidden',
              chat.isExpanded
                ? 'w-[90vw] max-w-[640px] h-[82vh] max-h-[740px]'
                : 'w-[400px] h-[560px]'
            )}
            role="dialog"
            aria-label="AI Chat Assistant"
          >
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-dark-800 bg-dark-900">
              <div className="flex items-center gap-3">
                <div className="relative">
                  <div className="w-8 h-8 rounded-lg bg-dark-800 border border-dark-700 flex items-center justify-center">
                    <img src="/brain-slug.svg" alt="" className="w-5 h-5" />
                  </div>
                  <span className="absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 bg-emerald-500 rounded-full border-[1.5px] border-dark-900" />
                </div>
                <div>
                  <h2 className="font-semibold text-white text-sm leading-tight">MidLens AI</h2>
                  <p className="text-[11px] text-dark-500">
                    {chat.currentAnalysis ? 'Analysis context active' : 'Medical assistant'}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-0.5">
                <HeaderBtn icon={ArrowsPointingOutIcon} label="Expand" onClick={() => chat.toggle('isExpanded')} />
                <HeaderBtn icon={MinusIcon} label="Minimize" onClick={() => chat.toggle('isMinimized')} />
                <HeaderBtn icon={ArrowPathIcon} label="Clear" onClick={chat.clearChat} />
                <HeaderBtn icon={XMarkIcon} label="Close" onClick={() => chat.toggle('isOpen')} />
              </div>
            </div>

            {/* Analysis context pill */}
            {chat.currentAnalysis && (
              <div className="px-4 py-2 border-b border-dark-800 bg-dark-900/80">
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-500/8 border border-emerald-500/15">
                  <SparklesIcon className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                  <span className="text-xs text-emerald-300 truncate">
                    {chat.currentAnalysis.prediction.display_name}
                  </span>
                  <span className="text-[10px] text-emerald-500/60 ml-auto shrink-0">
                    {(chat.currentAnalysis.prediction.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            )}

            {/* Messages Area */}
            <div
              ref={messagesRef}
              className="flex-1 overflow-y-auto p-4 space-y-5 scrollbar-thin"
            >
              {chat.messages.length === 0 ? (
                <WelcomeScreen
                  suggestions={chat.suggestions}
                  onSuggestionClick={s => chat.sendMessage(s)}
                  hasAnalysis={!!chat.currentAnalysis}
                />
              ) : (
                <>
                  {chat.messages.map(msg => (
                    <MessageBubble key={msg.id} message={msg} />
                  ))}
                  {chat.isLoading && <TypingIndicator />}
                </>
              )}
            </div>

            {/* Input Area */}
            <form onSubmit={handleSubmit} className="p-3 border-t border-dark-800 bg-dark-900/80">
              <div className="flex items-end gap-2 bg-dark-800 rounded-xl border border-dark-700/50 px-3 py-2 focus-within:border-primary-500/40 transition-colors">
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={handleTextareaResize}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask about brain tumors..."
                  disabled={chat.isLoading}
                  rows={1}
                  className={cn(
                    'flex-1 bg-transparent text-sm text-white placeholder-dark-500',
                    'resize-none outline-none py-1.5',
                    'disabled:opacity-40 max-h-[120px]'
                  )}
                />
                <button
                  type="submit"
                  disabled={!input.trim() || chat.isLoading}
                  className={cn(
                    'shrink-0 p-2 rounded-lg transition-all',
                    input.trim()
                      ? 'bg-primary-600 text-white hover:bg-primary-500'
                      : 'text-dark-600 cursor-not-allowed'
                  )}
                >
                  <PaperAirplaneIcon className="w-4 h-4" />
                </button>
              </div>
              <p className="text-[10px] text-dark-600 text-center mt-2">
                AI responses are informational only — always consult a doctor
              </p>
            </form>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Minimized Pill ── */}
      <AnimatePresence>
        {chat.isOpen && chat.isMinimized && (
          <motion.button
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            onClick={() => chat.toggle('isMinimized')}
            className="fixed bottom-6 right-6 z-50 flex items-center gap-2.5 px-4 py-2.5 rounded-xl bg-dark-800 border border-dark-700 text-white shadow-xl hover:border-dark-600 transition-colors"
          >
            <div className="w-5 h-5 rounded-md bg-dark-700 flex items-center justify-center">
              <img src="/brain-slug.svg" alt="" className="w-3.5 h-3.5" />
            </div>
            <span className="text-sm font-medium">MidLens AI</span>
            {chat.messages.length > 0 && (
              <span className="px-1.5 py-0.5 rounded-md bg-dark-700 text-[11px] text-dark-300">
                {chat.messages.length}
              </span>
            )}
          </motion.button>
        )}
      </AnimatePresence>
    </>
  )
}

// ─── Header Button ──────────────────────────────────
const HeaderBtn = memo(function HeaderBtn({
  icon: Icon,
  label,
  onClick,
}: {
  icon: React.ElementType
  label: string
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className="p-1.5 rounded-lg text-dark-500 hover:text-dark-200 hover:bg-dark-800 transition-colors"
      aria-label={label}
    >
      <Icon className="w-4 h-4" />
    </button>
  )
})

// ─── Welcome Screen ─────────────────────────────────
const WelcomeScreen = memo(function WelcomeScreen({
  suggestions,
  onSuggestionClick,
  hasAnalysis,
}: {
  suggestions: string[]
  onSuggestionClick: (text: string) => void
  hasAnalysis: boolean
}) {
  return (
    <div className="h-full flex flex-col items-center justify-center px-2">
      <div className="w-12 h-12 rounded-xl bg-dark-800 border border-dark-700 flex items-center justify-center mb-4">
        <img src="/brain-slug.svg" alt="" className="w-7 h-7" />
      </div>
      <h3 className="text-base font-semibold text-white mb-1">How can I help?</h3>
      <p className="text-sm text-dark-500 text-center mb-8 max-w-[280px]">
        {hasAnalysis
          ? 'Ask me about your scan results or brain tumor information.'
          : 'Ask about brain tumors, symptoms, or treatment options.'}
      </p>

      <div className="w-full space-y-2">
        <p className="text-[11px] text-dark-600 flex items-center gap-1.5 mb-2 px-1">
          <LightBulbIcon className="w-3.5 h-3.5" />
          Try asking
        </p>
        {suggestions.map((s, i) => (
          <button
            key={i}
            onClick={() => onSuggestionClick(s)}
            className={cn(
              'w-full px-4 py-3 text-left text-sm rounded-xl',
              'bg-dark-800/60 border border-dark-700/40',
              'text-dark-300 hover:text-white',
              'hover:bg-dark-800 hover:border-dark-600',
              'transition-all duration-150'
            )}
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  )
})

// ─── Message Bubble ─────────────────────────────────
const MessageBubble = memo(function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user'

  if (message.role === 'system') {
    return (
      <div className="flex justify-center">
        <div className="px-3 py-1 rounded-full bg-dark-800 border border-dark-700 text-[11px] text-dark-400">
          {message.content}
        </div>
      </div>
    )
  }

  return (
    <div className={cn('flex gap-2.5', isUser ? 'flex-row-reverse' : 'flex-row')}>
      {/* Avatar */}
      {!isUser && (
        <div className="shrink-0 mt-0.5">
          <div className="w-7 h-7 rounded-lg bg-dark-800 border border-dark-700 flex items-center justify-center">
            <img src="/brain-slug.svg" alt="" className="w-4 h-4" />
          </div>
        </div>
      )}

      <div className={cn('max-w-[82%] space-y-2', isUser && 'flex flex-col items-end')}>
        {/* Bubble */}
        <div
          className={cn(
            'px-3.5 py-2.5 text-sm leading-relaxed',
            isUser
              ? 'bg-primary-600 text-white rounded-2xl rounded-br-md'
              : message.error
                ? 'bg-red-950/40 text-red-200 border border-red-800/30 rounded-2xl rounded-bl-md'
                : 'bg-dark-800 text-dark-200 border border-dark-700/50 rounded-2xl rounded-bl-md'
          )}
        >
          <FormattedContent content={message.content} />
        </div>

        {/* Sources & Confidence (assistant only) */}
        {!isUser && !message.error && (message.sources?.length || message.confidence) && (
          <SourcesPanel sources={message.sources} confidence={message.confidence} />
        )}
      </div>
    </div>
  )
})

// ─── Sources Panel (Expandable) ─────────────────────
function SourcesPanel({
  sources,
  confidence,
}: {
  sources?: Source[]
  confidence?: number
}) {
  const [expanded, setExpanded] = useState(false)
  const hasSources = sources && sources.length > 0
  const isVerified = hasSources && confidence && confidence >= 0.7

  const handleToggle = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    setExpanded(prev => !prev)
  }, [])

  return (
    <div className="w-full relative z-10">
      {/* Toggle row */}
      <div
        role="button"
        tabIndex={0}
        onClick={handleToggle}
        onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handleToggle(e as unknown as React.MouseEvent) } }}
        className="flex items-center gap-2 px-2 py-1.5 text-[11px] group cursor-pointer rounded-lg hover:bg-dark-800/40 transition-colors select-none"
      >
        {isVerified ? (
          <span className="flex items-center gap-1 text-emerald-400">
            <CheckBadgeIcon className="w-3.5 h-3.5" />
            Verified
          </span>
        ) : hasSources ? (
          <span className="flex items-center gap-1 text-dark-500">
            <DocumentTextIcon className="w-3.5 h-3.5" />
            Sources available
          </span>
        ) : (
          <span className="flex items-center gap-1 text-amber-500/70">
            <ExclamationTriangleIcon className="w-3.5 h-3.5" />
            No sources
          </span>
        )}

        {hasSources && (
          <>
            <span className="text-dark-600">·</span>
            <span className="text-dark-500 group-hover:text-dark-300 transition-colors">
              {sources.length} {sources.length === 1 ? 'source' : 'sources'}
            </span>
            {expanded ? (
              <ChevronUpIcon className="w-3 h-3 text-dark-500 ml-auto" />
            ) : (
              <ChevronDownIcon className="w-3 h-3 text-dark-500 ml-auto" />
            )}
          </>
        )}
      </div>

      {/* Expanded source list */}
      {expanded && hasSources && (
        <div className="mt-1 space-y-1.5 pl-1 animate-in fade-in slide-in-from-top-1 duration-200">
          {sources.map((src, i) => (
            <div
              key={i}
              className="flex items-start gap-2 px-3 py-2 rounded-lg bg-dark-800/60 border border-dark-700/30"
            >
              <DocumentTextIcon className="w-3.5 h-3.5 text-dark-500 mt-0.5 shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-xs text-dark-300 truncate">{src.title}</p>
                <div className="flex items-center gap-2 mt-0.5">
                  <span className="text-[10px] text-dark-600 capitalize">
                    {src.source_type.replace('_', ' ')}
                  </span>
                  <ReliabilityDot score={src.reliability} />
                </div>
              </div>
            </div>
          ))}

          {confidence !== undefined && (
            <div className="flex items-center gap-2 px-3 py-1.5 text-[10px] text-dark-500">
              <span>Overall confidence:</span>
              <span
                className={cn(
                  'font-medium',
                  confidence >= 0.8 ? 'text-emerald-400' : confidence >= 0.6 ? 'text-amber-400' : 'text-red-400'
                )}
              >
                {(confidence * 100).toFixed(0)}%
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ─── Reliability Dot ────────────────────────────────
function ReliabilityDot({ score }: { score: number }) {
  const color = score >= 0.8 ? 'bg-emerald-500' : score >= 0.5 ? 'bg-amber-500' : 'bg-red-500'
  const label = score >= 0.8 ? 'High' : score >= 0.5 ? 'Medium' : 'Low'
  return (
    <span className="flex items-center gap-1 text-[10px] text-dark-500">
      <span className={cn('w-1.5 h-1.5 rounded-full', color)} />
      {label}
    </span>
  )
}

// ─── Formatted Content ──────────────────────────────
const FormattedContent = memo(function FormattedContent({ content }: { content: string }) {
  const html = useMemo(() => {
    if (!content) return '<span class="text-dark-500 italic">No content</span>'

    return content
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/\*\*(.*?)\*\*/g, '<strong class="text-white font-medium">$1</strong>')
      .replace(/^### (.*$)/gm, '<h4 class="font-semibold text-white mt-3 mb-1 text-[13px]">$1</h4>')
      .replace(/^## (.*$)/gm, '<h3 class="font-semibold text-white mt-3 mb-2">$1</h3>')
      .replace(/^[•\-] (.*$)/gm, '<li class="ml-3 list-disc list-inside">$1</li>')
      .replace(/^(\d+)\. (.*$)/gm, '<li class="ml-3"><span class="text-primary-400 font-medium">$1.</span> $2</li>')
      .replace(/\n{2,}/g, '<br/><br/>')
      .replace(/\n/g, '<br/>')
  }, [content])

  return (
    <div
      className="prose prose-sm prose-invert max-w-none leading-relaxed [&_li]:my-0.5"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
})

// ─── Typing Indicator ───────────────────────────────
const TypingIndicator = memo(function TypingIndicator() {
  return (
    <div className="flex gap-2.5">
      <div className="shrink-0">
        <div className="w-7 h-7 rounded-lg bg-dark-800 border border-dark-700 flex items-center justify-center">
          <img src="/brain-slug.svg" alt="" className="w-4 h-4" />
        </div>
      </div>
      <div className="px-4 py-3 rounded-2xl rounded-bl-md bg-dark-800 border border-dark-700/50">
        <div className="flex gap-1.5">
          {[0, 1, 2].map(i => (
            <span
              key={i}
              className="w-1.5 h-1.5 bg-dark-400 rounded-full animate-bounce"
              style={{ animationDelay: `${i * 0.15}s`, animationDuration: '0.8s' }}
            />
          ))}
        </div>
      </div>
    </div>
  )
})

export default ChatbotV2
