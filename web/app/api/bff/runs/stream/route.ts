import { NextRequest } from 'next/server';

const ORCH = process.env.ORCHESTRATION_BASE || process.env.NEXT_PUBLIC_ORCHESTRATION_BASE || 'http://orchestration-service:8010';

export async function GET(_req: NextRequest) {
  const res = await fetch(`${ORCH}/runs/stream`, { cache: 'no-store' });
  const readable = res.body as any;
  return new Response(readable, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive'
    }
  });
}
