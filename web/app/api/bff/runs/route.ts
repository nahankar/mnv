import { NextResponse } from 'next/server';
const ORCH = process.env.ORCHESTRATION_BASE || process.env.NEXT_PUBLIC_ORCHESTRATION_BASE || 'http://orchestration-service:8010';

export async function GET() {
  const r = await fetch(`${ORCH}/runs`, { cache: 'no-store' });
  const j = await r.json();
  return NextResponse.json(j);
}
