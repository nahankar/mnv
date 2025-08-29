import { NextRequest, NextResponse } from 'next/server';
import crypto from 'crypto';
import { getWithETag, setCachedJSON, del } from '@/lib/cache';
import { auth } from '@/auth';

const ORCH = process.env.ORCHESTRATION_BASE || process.env.NEXT_PUBLIC_ORCHESTRATION_BASE || 'http://orchestration-service:8010';

export async function GET() {
  const cacheKey = 'budgets:all';
  const { etag, value } = await getWithETag(cacheKey);
  if (value && etag) {
    return NextResponse.json(value, { headers: { ETag: etag } });
  }
  const [b, c] = await Promise.all([
    fetch(`${ORCH}/budgets`, { cache: 'no-store' }).then(r => r.json()),
    fetch(`${ORCH}/costs?user_id=admin`, { cache: 'no-store' }).then(r => r.json())
  ]);
  const payload = { budgets: b.budgets || {}, costs: c.costs || {} };
  const newTag = 'W/"' + crypto.createHash('sha1').update(JSON.stringify(payload)).digest('hex') + '"';
  await setCachedJSON(cacheKey, payload, 30, newTag);
  return NextResponse.json(payload, { headers: { ETag: newTag } });
}

export async function POST(req: NextRequest) {
  const session = await auth();
  const role = (session as any)?.user?.role || 'viewer';
  if (role !== 'admin') {
    return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
  }
  const csrf = req.headers.get('x-csrf-token');
  if (!csrf || csrf !== process.env.CSRF_TOKEN) {
    return NextResponse.json({ error: 'CSRF' }, { status: 403 });
  }
  const body = await req.json();
  const r = await fetch(`${ORCH}/budgets`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json();
  await del('budgets:all');
  return NextResponse.json(j, { status: r.status });
}
