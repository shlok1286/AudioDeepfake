'use client'

import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Menu, X } from 'lucide-react'

interface NavbarProps {
  activeTab: 'home' | 'history'
  onSelectTab: (tab: 'home' | 'history') => void
  onScrollToSection: (sectionId: string) => void
}

type NavSection = 'home' | 'how-it-works' | 'history' | 'faq' | 'contact'

export default function Navbar({ activeTab, onSelectTab, onScrollToSection }: NavbarProps) {
  const [activeSection, setActiveSection] = useState<NavSection>('home')
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  // Synchronize active section with scroll position
  useEffect(() => {
    if (activeTab === 'history') {
      setActiveSection('history')
      return
    }

    const handleScroll = () => {
      const scrollY = window.scrollY
      const sections: { id: NavSection; el: HTMLElement | null }[] = [
        { id: 'contact', el: document.getElementById('contact') },
        { id: 'faq', el: document.getElementById('faq') },
        { id: 'how-it-works', el: document.getElementById('how-it-works') },
        { id: 'home', el: document.getElementById('top') },
      ]

      for (const section of sections) {
        if (section.el) {
          const top = section.el.offsetTop - 140
          if (scrollY >= top) {
            setActiveSection(section.id)
            return
          }
        }
      }
      setActiveSection('home')
    }

    window.addEventListener('scroll', handleScroll, { passive: true })
    handleScroll()
    return () => window.removeEventListener('scroll', handleScroll)
  }, [activeTab])

  const handleNavClick = (section: NavSection) => {
    setMobileMenuOpen(false)
    if (section === 'history') {
      setActiveSection('history')
      onSelectTab('history')
      window.scrollTo({ top: 0, behavior: 'smooth' })
    } else {
      setActiveSection(section)
      if (activeTab !== 'home') {
        onSelectTab('home')
      }
      const targetId = section === 'home' ? 'top' : section
      onScrollToSection(targetId)
    }
  }

  const navItems: { label: string; section: NavSection }[] = [
    { label: 'Home', section: 'home' },
    { label: 'How It Works', section: 'how-it-works' },
    { label: 'History', section: 'history' },
    { label: 'FAQ', section: 'faq' },
    { label: 'Contact', section: 'contact' },
  ]

  return (
    <motion.header
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: 'easeOut' }}
      className="sticky top-3 sm:top-5 z-40 mx-auto w-[92%] sm:w-fit max-w-[860px]"
    >
      <div className="relative flex h-[64px] sm:h-[68px] items-center justify-between gap-6 sm:gap-10 rounded-full border border-[rgba(156,188,255,0.15)] bg-[#070b18]/85 px-5 sm:px-7 shadow-[0_8px_32px_rgba(0,0,0,0.5),0_0_20px_rgba(23,103,255,0.06)] backdrop-blur-xl transition-all">
        {/* LEFT: ONLY UPLOADED LOGO AS BRAND MARK (NO DUPLICATE TEXT) */}
        <button
          onClick={() => handleNavClick('home')}
          className="group flex items-center text-left focus:outline-none cursor-pointer select-none shrink-0"
          aria-label="VerifyVoice Home"
        >
          <img
            src="/verifyvoice-logo.png"
            onError={(e) => {
              e.currentTarget.src = '/verifyvoice-logo.jpg'
            }}
            alt="VerifyVoice"
            className="h-[36px] sm:h-[38px] w-auto max-h-[40px] object-contain transition-transform duration-200 group-hover:scale-105 drop-shadow-[0_0_10px_rgba(37,99,235,0.45)]"
          />
        </button>

        {/* NAVIGATION ITEMS (CENTER / RIGHT) */}
        <nav className="hidden md:flex items-center gap-1.5 lg:gap-2.5" aria-label="Main Navigation">
          {navItems.map((item) => {
            const isActive = activeSection === item.section
            return (
              <button
                key={item.section}
                onClick={() => handleNavClick(item.section)}
                className={`relative px-4 py-1.5 text-[13.5px] lg:text-[14px] font-medium transition-all duration-200 rounded-full cursor-pointer whitespace-nowrap ${
                  isActive
                    ? 'bg-[rgba(23,103,255,0.14)] text-[#64adff] border border-[rgba(100,173,255,0.3)] shadow-[0_0_12px_rgba(23,103,255,0.18)]'
                    : 'text-[#8b98b5] hover:text-[#eef4ff] hover:bg-[rgba(255,255,255,0.04)] border border-transparent'
                }`}
              >
                {item.label}
              </button>
            )
          })}
        </nav>

        {/* MOBILE: HAMBURGER TOGGLE */}
        <div className="flex md:hidden items-center">
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            aria-label={mobileMenuOpen ? 'Close Menu' : 'Open Menu'}
            className="flex size-10 items-center justify-center rounded-full border border-[rgba(156,188,255,0.15)] bg-[rgba(16,24,43,0.6)] text-[#8b98b5] hover:border-[rgba(100,173,255,0.3)] hover:text-[#eef4ff] transition-colors cursor-pointer"
          >
            {mobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>

        {/* MOBILE DROPDOWN PANEL */}
        <AnimatePresence>
          {mobileMenuOpen && (
            <motion.div
              initial={{ opacity: 0, y: -8, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -8, scale: 0.98 }}
              transition={{ duration: 0.2, ease: 'easeOut' }}
              className="absolute left-0 right-0 top-[calc(100%+10px)] flex flex-col rounded-3xl border border-[rgba(156,188,255,0.18)] bg-[#070b18]/95 p-4 shadow-[0_16px_40px_rgba(0,0,0,0.6),0_0_24px_rgba(23,103,255,0.1)] backdrop-blur-2xl md:hidden z-50"
            >
              <div className="flex flex-col gap-1.5">
                {navItems.map((item) => {
                  const isActive = activeSection === item.section
                  return (
                    <button
                      key={item.section}
                      onClick={() => handleNavClick(item.section)}
                      className={`flex w-full items-center px-4 py-2.5 text-left text-[14px] rounded-xl transition-all cursor-pointer ${
                        isActive
                          ? 'bg-[rgba(23,103,255,0.16)] text-[#64adff] border border-[rgba(100,173,255,0.3)] font-medium shadow-[0_0_12px_rgba(23,103,255,0.15)]'
                          : 'font-medium text-[#8b98b5] hover:bg-[rgba(255,255,255,0.04)] hover:text-[#eef4ff]'
                      }`}
                    >
                      {item.label}
                    </button>
                  )
                })}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.header>
  )
}
