import { lazy, StrictMode, Suspense } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './app/App';
import './app/styles.css';

const DiagnosticsApp = lazy(() => import('./app/DiagnosticsApp'));

const query = new URLSearchParams(window.location.search);
const diagnostics =
  window.location.pathname.startsWith('/diagnostics') || query.has('diagnostics') || query.has('frames');

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    {diagnostics ? (
      <Suspense fallback={null}>
        <DiagnosticsApp />
      </Suspense>
    ) : (
      <App />
    )}
  </StrictMode>,
);
