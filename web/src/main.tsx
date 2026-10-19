import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './style.css';

// `npm run dev:mock` serves red-team routes from src/mocks; everything else still hits the backend.
async function start() {
  if (import.meta.env.DEV && import.meta.env.VITE_API_MOCK === '1') {
    const { worker } = await import('./mocks/browser');
    await worker.start({ onUnhandledRequest: 'bypass' });
  }
  ReactDOM.createRoot(document.getElementById('root')!).render(<React.StrictMode><App /></React.StrictMode>);
}
start();
