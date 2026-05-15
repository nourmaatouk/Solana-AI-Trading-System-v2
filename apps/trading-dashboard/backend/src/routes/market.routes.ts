import { Router, Request, Response } from 'express';
import { BinanceWebSocketService } from '../services/binanceWebSocket.service';

export function marketRouter(binanceService: BinanceWebSocketService): Router {
  const router = Router();

  router.get('/status', (_req: Request, res: Response) => {
    res.json({
      connection: binanceService.getConnectionStatus(),
      exchange: binanceService.getExchange(),
      ticker: binanceService.getLatestTicker(),
      depth: binanceService.getLatestDepth(),
    });
  });

  router.get('/ticker', (_req: Request, res: Response) => {
    const ticker = binanceService.getLatestTicker();
    if (!ticker) {
      res.status(404).json({ error: 'No ticker data yet' });
      return;
    }
    res.json(ticker);
  });

  router.get('/depth', (_req: Request, res: Response) => {
    const depth = binanceService.getLatestDepth();
    if (!depth) {
      res.status(404).json({ error: 'No depth data yet' });
      return;
    }
    res.json(depth);
  });

  router.get('/klines', (_req: Request, res: Response) => {
    res.json(binanceService.getLatestKlines());
  });

  router.get('/trades', (_req: Request, res: Response) => {
    res.json(binanceService.getLatestTrades());
  });

  return router;
}
