import { NextRequest, NextResponse } from 'next/server';
import { getWithETag, setCachedJSON } from '@/lib/cache';
import crypto from 'crypto';
import { auth } from '@/auth';

const MOD = process.env.MODERATION_BASE || process.env.NEXT_PUBLIC_MODERATION_BASE || 'http://moderation-service:8006';

export async function GET() {
  const cacheKey = 'review:queue';
  const { etag, value } = await getWithETag(cacheKey);
  if (etag && value) return NextResponse.json(value, { headers: { ETag: etag } });
  const r = await fetch(`${MOD}/review/queue?limit=50&offset=0`, { cache: 'no-store' });
  const j = await r.json();
  const newTag = 'W/"' + crypto.createHash('sha1').update(JSON.stringify(j)).digest('hex') + '"';
  await setCachedJSON(cacheKey, j, 15, newTag);
  return NextResponse.json(j, { headers: { ETag: newTag } });
}

export async function POST(req: NextRequest) {
  const session = await auth();
  const role = (session as any)?.user?.role || 'viewer';
  if (!['operator', 'admin'].includes(role)) return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
  const body = await req.json();
  const { moderation_id, decision, notes, override_reason } = body || {};
  const r = await fetch(`${MOD}/review/${moderation_id}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ moderation_id, reviewer_id: (session as any)?.user?.id || 'op', decision, notes, override_reason })
  });
  const j = await r.json().catch(() => ({}));
  return NextResponse.json(j, { status: r.status });
}
