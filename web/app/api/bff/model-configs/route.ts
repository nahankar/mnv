import { NextRequest, NextResponse } from 'next/server';
import crypto from 'crypto';
import { getWithETag, setCachedJSON, del } from '@/lib/cache';

const ORCH = process.env.ORCHESTRATION_BASE || process.env.NEXT_PUBLIC_ORCHESTRATION_BASE || 'http://orchestration-service:8010';

export async function GET(req: NextRequest) {
  const url = new URL(req.url);
  const type = url.searchParams.get('config_type');
  const cacheKey = `mc:${type || 'all'}`;
  const { etag, value } = await getWithETag(cacheKey);
  const r = await fetch(`${ORCH}/model-configs${type ? `?config_type=${encodeURIComponent(type)}` : ''}`, { cache: 'no-store', headers: etag ? { 'If-None-Match': etag } : {} });
  if (r.status === 304 && value) {
    return NextResponse.json(value, { status: 200, headers: { ETag: etag! } });
  }
  const j = await r.json();
  const newTag = 'W/"' + crypto.createHash('sha1').update(JSON.stringify(j)).digest('hex') + '"';
  await setCachedJSON(cacheKey, j, 60, newTag);
  return NextResponse.json(j, { status: r.status, headers: { ETag: newTag } });
}

export async function POST(req: NextRequest) {
  const body = await req.json();
  const r = await fetch(`${ORCH}/model-configs`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json();
  // cache bust
  await del('mc:all');
  await del(`mc:${body?.config_type || ''}`);
  return NextResponse.json(j, { status: r.status });
}
