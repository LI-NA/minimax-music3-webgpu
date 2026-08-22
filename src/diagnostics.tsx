import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import DiagnosticsApp from './app/DiagnosticsApp';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <DiagnosticsApp />
  </StrictMode>,
);
