"use client";

import useSWR from 'swr';
import { Button } from '@/components/ui/button';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Spinner, notify } from '@/components/ui';

export default function ReviewQueuePage() {
  const { data, isLoading, mutate } = useSWR('/api/bff/review', (url) => fetch(url).then(r => r.json()));
  const items = data?.items || [];

  async function takeAction(id: string, decision: string) {
    const r = await fetch('/api/bff/review', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ moderation_id: id, decision, notes: '', override_reason: '' })
    });
    if (!r.ok) { notify.error('Action failed'); return; }
    notify.success('Action recorded');
    await mutate();
  }

  return (
    <main>
      <h1 className="text-xl font-semibold mb-4">Review Queue</h1>
      {isLoading ? (
        <div className="text-gray-500 flex items-center gap-2"><Spinner /> <span>Loading...</span></div>
      ) : (
        <Card>
          <CardHeader><CardTitle>Pending Items</CardTitle></CardHeader>
          <CardContent>
            <table className="min-w-full text-sm">
              <thead><tr className="text-left"><th className="p-2">ID</th><th className="p-2">Type</th><th className="p-2">Score</th><th className="p-2">Status</th><th className="p-2">Actions</th></tr></thead>
              <tbody>
                {items.map((it: any) => (
                  <tr key={it.id} className="border-t">
                    <td className="p-2">{it.id}</td>
                    <td className="p-2">{it.content_type}</td>
                    <td className="p-2">{it.score}</td>
                    <td className="p-2">{it.status}</td>
                    <td className="p-2 flex gap-2">
                      <Button size="sm" onClick={() => takeAction(it.id, 'approved')}>Approve</Button>
                      <Button size="sm" variant="destructive" onClick={() => takeAction(it.id, 'rejected')}>Reject</Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>
      )}
    </main>
  );
}
