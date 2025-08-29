"use client";

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { notify } from '@/components/ui';

export default function TriggersPage() {
  const [output, setOutput] = useState('');
  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const fd = new FormData(e.currentTarget);
    const genre = (fd.get('genre') as string) || 'fantasy';
    const theme = (fd.get('theme') as string) || 'adventure';
    const target_length = Number(fd.get('target_length') || 200);
    const base = process.env.NEXT_PUBLIC_ORCHESTRATION_BASE || 'http://localhost:8010';
    const r = await fetch(`${base}/run/story-to-video?genre=${encodeURIComponent(genre)}&theme=${encodeURIComponent(theme)}&target_length=${target_length}`, { method: 'POST' });
    const j = await r.json().catch(() => ({}));
    setOutput(JSON.stringify(j, null, 2));
    if (r.ok) notify.success('Flow triggered'); else notify.error('Trigger failed');
  }

  return (
    <main>
      <h1 className="text-xl font-semibold mb-4">Triggers</h1>
      <Card>
        <CardHeader><CardTitle>Story to Video</CardTitle></CardHeader>
        <CardContent>
          <form onSubmit={onSubmit} className="grid gap-2 max-w-md">
            <label>Genre<input className="border p-1" name="genre" defaultValue="fantasy" /></label>
            <label>Theme<input className="border p-1" name="theme" defaultValue="adventure" /></label>
            <label>Target Length<input className="border p-1" type="number" name="target_length" defaultValue={200} /></label>
            <Button type="submit">Run story-to-video</Button>
          </form>
          {output && (
            <pre className="mt-4 bg-gray-900 text-green-400 p-3 rounded max-w-3xl overflow-auto">{output}</pre>
          )}
        </CardContent>
      </Card>
    </main>
  );
}
