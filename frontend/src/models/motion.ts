// Viewers who ask the system for reduced motion get still views: no automatic turning or smooth scrolling.
export const prefersReducedMotion = () =>
  typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches === true
