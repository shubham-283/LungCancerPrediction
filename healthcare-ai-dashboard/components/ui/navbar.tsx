// components/Header.tsx
'use client';

import React, { useEffect, useRef, useState } from 'react';

export default function Header() {
  const [visible, setVisible] = useState(true);
  const lastY = useRef(0);
  const rafId = useRef<number | null>(null);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    lastY.current = window.scrollY;

    const onScroll = () => {
      if (rafId.current !== null) return;
      rafId.current = window.requestAnimationFrame(() => {
        const y = window.scrollY;
        const delta = y - lastY.current;

        const scrollingDown = delta > 0;
        const smallUp = delta < -2; // show on small upward scroll
        const pastTop = y > 8; // avoid hiding immediately at the top

        if (scrollingDown && pastTop) {
          setVisible(false);
        } else if (smallUp) {
          setVisible(true);
        }

        lastY.current = y;
        rafId.current = null;
      });
    };

    window.addEventListener('scroll', onScroll, { passive: true });

    return () => {
      if (rafId.current !== null) {
        cancelAnimationFrame(rafId.current);
        rafId.current = null;
      }
      window.removeEventListener('scroll', onScroll);
    };
  }, []);

  const scrollToSection = (id: string) => {
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  return (
    <header
      className={`bg-white/90 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-50 transition-transform duration-200 ease-in-out will-change-transform ${
        visible ? 'translate-y-0' : '-translate-y-full'
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-sm">AI</span>
              </div>
              <span className="text-xl font-bold text-gray-900">
                LungAI Diagnostic System
              </span>
            </div>
          </div>

          <nav className="hidden md:flex items-center space-x-8">
            <button
              onClick={() => scrollToSection('patient-input')}
              className="text-gray-600 hover:text-blue-600 font-medium transition-colors"
            >
              Patient Assessment
            </button>
            <button
              onClick={() => scrollToSection('ct-scan-analysis')}
              className="text-gray-600 hover:text-blue-600 font-medium transition-colors"
            >
              CT Scan Analysis
            </button>
          </nav>
        </div>
      </div>
    </header>
  );
}
