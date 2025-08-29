"use client";

import { useEffect, useState } from 'react';

export default function RunsPage() {
  const [items, setItems] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  async function load() {
    setLoading(true);
    try {
      const r = await fetch('/api/bff/runs', { cache: 'no-store' });
      const j = await r.json();
      setItems(j.items || []);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    const ev = new EventSource('/api/bff/runs/stream');
    ev.onmessage = (e) => {
      try {
        const arr = JSON.parse(e.data);
        if (Array.isArray(arr)) setItems(arr);
      } catch {}
    };
    return () => ev.close();
  }, []);

  return (
    <main>
      <h1 className="text-xl font-semibold mb-4">Recent Pipeline Runs</h1>
      {loading && <p className="text-gray-500">Loading...</p>}
      <table className="min-w-full text-sm">
        <thead><tr className="text-left"><th className="p-2">Run ID</th><th className="p-2">Flow</th><th className="p-2">Status</th><th className="p-2">Started</th></tr></thead>
        <tbody>
          {items.map((it) => (
            <tr key={it.id} className="border-t">
              <td className="p-2">{it.id}</td>
              <td className="p-2">{it.flow}</td>
              <td className="p-2">{it.status}</td>
              <td className="p-2">{it.created_at}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </main>
  );
}
