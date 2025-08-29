import { NextRequest, NextResponse } from 'next/server';
import { del } from '@/lib/cache';

const ORCH = process.env.ORCHESTRATION_BASE || process.env.NEXT_PUBLIC_ORCHESTRATION_BASE || 'http://orchestration-service:8010';

export async function PUT(req: NextRequest, { params }: { params: { id: string } }) {
  const body = await req.json();
  const r = await fetch(`${ORCH}/model-configs/${params.id}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({}));
  await del('mc:all');
  await del(`mc:${body?.config_type || ''}`);
  return NextResponse.json(j, { status: r.status });
}

export async function DELETE(_req: NextRequest, { params }: { params: { id: string } }) {
  const r = await fetch(`${ORCH}/model-configs/${params.id}`, { method: 'DELETE' });
  const j = await r.json().catch(() => ({}));
  await del('mc:all');
  return NextResponse.json(j, { status: r.status });
}
