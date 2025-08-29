"use client";

import { useEffect, useState } from 'react';
import useSWR from 'swr';
import { notify } from '@/components/ui';
import { Button } from '@/components/ui/button';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { DangerButton, Spinner } from '@/components/ui';

export default function ModelConfigsPage() {
  const [items, setItems] = useState<any[]>([]);
  const [form, setForm] = useState<any>({ name: '', config_type: 'story', version: 'v1', parameters: '{}' });

  const { data, error, mutate, isLoading } = useSWR('/api/bff/model-configs', (url) => fetch(url).then(r => r.json()));
  useEffect(() => { setItems(data?.items || []); }, [data]);

  async function create(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const payload = {
      name: form.name,
      description: form.description || '',
      config_type: form.config_type,
      provider: form.provider || '',
      model_name: form.model_name || '',
      parameters: JSON.parse(form.parameters || '{}'),
      cost_per_unit: form.cost_per_unit ? Number(form.cost_per_unit) : undefined,
      performance_metrics: {},
      version: form.version,
      is_active: true,
      is_default: false,
      ab_test_group: form.ab_test_group || '',
      traffic_percentage: form.traffic_percentage ? Number(form.traffic_percentage) : 100,
    };
    const r = await fetch('/api/bff/model-configs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    if (!r.ok) { notify.error('Create failed'); return; }
    notify.success('Created');
    setForm({ name: '', config_type: 'story', version: 'v1', parameters: '{}' });
    await mutate();
  }

  async function del(id: string) {
    const r = await fetch(`/api/bff/model-configs/${id}`, { method: 'DELETE' });
    if (!r.ok) { notify.error('Delete failed'); return; }
    notify.success('Deleted');
    await mutate();
  }

  return (
    <main>
      <h1 className="text-xl font-semibold mb-4">Model Configurations</h1>
      <form onSubmit={create} className="grid gap-2 max-w-3xl">
        <label>Name<input className="border p-1" value={form.name} onChange={e => setForm({ ...form, name: e.target.value })} required /></label>
        <label>Type<select className="border p-1" value={form.config_type} onChange={e => setForm({ ...form, config_type: e.target.value })}>
          <option value="story">story</option>
          <option value="tts">tts</option>
          <option value="image">image</option>
          <option value="music">music</option>
          <option value="video">video</option>
        </select></label>
        <label>Provider<input className="border p-1" value={form.provider || ''} onChange={e => setForm({ ...form, provider: e.target.value })} /></label>
        <label>Model Name<input className="border p-1" value={form.model_name || ''} onChange={e => setForm({ ...form, model_name: e.target.value })} /></label>
        <label>Version<input className="border p-1" value={form.version} onChange={e => setForm({ ...form, version: e.target.value })} required /></label>
        <label>Parameters (JSON)<textarea className="border p-1" value={form.parameters} onChange={e => setForm({ ...form, parameters: e.target.value })} rows={4} /></label>
        <label>Traffic %<input className="border p-1" type="number" value={form.traffic_percentage || 100} onChange={e => setForm({ ...form, traffic_percentage: e.target.value })} /></label>
        <Button disabled={isLoading} type="submit">{isLoading ? 'Saving...' : 'Create'}</Button>
      </form>

      <h2 className="mt-6 text-lg font-medium">Existing</h2>
      <table className="min-w-full text-sm">
        <thead><tr className="text-left"><th className="p-2">Name</th><th className="p-2">Type</th><th className="p-2">Version</th><th className="p-2">Provider</th><th className="p-2">Actions</th></tr></thead>
        <tbody>
          {items.map((it) => (
            <tr key={it.id} className="border-t">
              <td className="p-2">{it.name}</td><td className="p-2">{it.config_type}</td><td className="p-2">{it.version}</td><td className="p-2">{it.provider}</td>
              <td className="p-2"><DangerButton onClick={() => del(it.id)}>Delete</DangerButton></td>
            </tr>
          ))}
        </tbody>
      </table>
    </main>
  );
}
