"use client";

import { useEffect, useState } from 'react';
import useSWR from 'swr';
import { notify } from '@/components/ui';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Spinner } from '@/components/ui';

export default function BudgetsPage() {
  const [form, setForm] = useState('');
  const { data, error, isLoading, mutate } = useSWR('/api/bff/budgets', (url) => fetch(url).then(r => r.json()));

  async function save() {
    try {
      const payload = { budgets: JSON.parse(form || '{}') };
      const r = await fetch('/api/bff/budgets', { method: 'POST', headers: { 'Content-Type': 'application/json', 'x-csrf-token': process.env.NEXT_PUBLIC_CSRF_TOKEN || '' }, body: JSON.stringify(payload) });
      if (!r.ok) { notify.error('Save failed'); return; }
      notify.success('Budgets updated');
      await mutate();
    } catch { notify.error('Invalid JSON'); }
  }

  return (
    <main>
      <h1 className="text-xl font-semibold mb-4">Budgets & Costs</h1>
      {isLoading && <div className="text-gray-500 flex items-center gap-2"><Spinner /> <span>Loading...</span></div>}
      <h2 className="mt-4 font-medium">Current Budgets</h2>
      <Card><pre>{JSON.stringify(data?.budgets || {}, null, 2)}</pre></Card>
      <h2 className="mt-4 font-medium">Today Costs</h2>
      <Card><pre>{JSON.stringify(data?.costs || {}, null, 2)}</pre></Card>
      <h2 className="mt-4 font-medium">Update Budgets (JSON)</h2>
      <textarea rows={6} value={form} onChange={e => setForm(e.target.value)} className="border p-2 w-[600px]" placeholder='{"story_generation": 5.0, "tts": 2.0}' />
      <div><Button className="mt-2" onClick={save}>Save</Button></div>
    </main>
  );
}
