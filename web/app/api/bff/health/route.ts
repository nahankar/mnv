import { NextResponse } from 'next/server'
import { getWithETag, setCachedJSON } from '@/lib/cache'
import crypto from 'crypto'

export const dynamic = 'force-dynamic'
export const runtime = 'nodejs'

const services: [string, string][] = [
  ['story', process.env.STORY_BASE || process.env.NEXT_PUBLIC_STORY_BASE || 'http://story-service:8001'],
  ['tts', process.env.TTS_BASE || process.env.NEXT_PUBLIC_TTS_BASE || 'http://tts-service:8002'],
  ['image', process.env.IMAGE_BASE || process.env.NEXT_PUBLIC_IMAGE_BASE || 'http://image-service:8003'],
  ['video', process.env.VIDEO_BASE || process.env.NEXT_PUBLIC_VIDEO_BASE || 'http://video-service:8005'],
  ['moderation', process.env.MODERATION_BASE || process.env.NEXT_PUBLIC_MODERATION_BASE || 'http://moderation-service:8006'],
  ['distribution', process.env.DISTRIBUTION_BASE || process.env.NEXT_PUBLIC_DISTRIBUTION_BASE || 'http://distribution-service:8007'],
  ['analytics', process.env.ANALYTICS_BASE || process.env.NEXT_PUBLIC_ANALYTICS_BASE || 'http://analytics-service:8008'],
  ['orchestration', process.env.ORCHESTRATION_BASE || process.env.NEXT_PUBLIC_ORCHESTRATION_BASE || 'http://orchestration-service:8010']
];

export async function GET() {
  const cacheKey = 'health:all';
  const { etag, value } = await getWithETag(cacheKey);
  if (value && etag) return NextResponse.json(value, { headers: { ETag: etag } });
  const results = await Promise.all(services.map(async ([name, base]) => {
    try {
      const r = await fetch(`${base}/health`, { cache: 'no-store' });
      const j = await r.json();
      return { name, status: j.status || 'unknown' };
    } catch (e) {
      return { name, status: 'error' };
    }
  }));
  const payload = { services: results };
  const newTag = 'W/"' + crypto.createHash('sha1').update(JSON.stringify(payload)).digest('hex') + '"';
  await setCachedJSON(cacheKey, payload, 10, newTag);
  return NextResponse.json(payload, { headers: { ETag: newTag } });
}
