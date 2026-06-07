import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { WalletProvider } from './providers/WalletProvider';
import { SystemWalletProvider } from './hooks/useSystemWallet';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <WalletProvider>
      <SystemWalletProvider>
        <App />
      </SystemWalletProvider>
    </WalletProvider>
  </React.StrictMode>,
);
