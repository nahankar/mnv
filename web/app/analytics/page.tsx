"use client";

import { useEffect, useState } from 'react';
import useSWR from 'swr';
import { notify } from '@/components/ui';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Spinner } from '@/components/ui';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

export default function AnalyticsPage() {
  const [platform, setPlatform] = useState('youtube');
  const [data, setData] = useState<any>(null);
  const base = process.env.NEXT_PUBLIC_ANALYTICS_BASE || 'http://localhost:8008';

  const { data: swrData, mutate, isLoading } = useSWR(`${base}/analytics/platform/${platform}`, (url) => fetch(url).then(r => r.json()));
  useEffect(() => { setData(swrData); }, [swrData]);

  async function collect() {
    try {
      const r = await fetch(`${base}/collect/${platform}`, { method: 'POST' });
      if (!r.ok) { notify.error('Collect failed'); return; }
      notify.success('Collection triggered');
      await mutate();
    } catch { notify.error('Collect error'); }
  }

  const views = (data?.items || []).map((x: any) => x.views || 0);
  const likes = (data?.items || []).map((x: any) => x.likes || 0);
  const labels = (data?.items || []).map((x: any, i: number) => x.date || i);
  const chartData = {
    labels,
    datasets: [
      { label: 'Views', data: views, borderColor: 'rgb(59,130,246)', backgroundColor: 'rgba(59,130,246,0.4)' },
      { label: 'Likes', data: likes, borderColor: 'rgb(34,197,94)', backgroundColor: 'rgba(34,197,94,0.4)' }
    ]
  };

  return (
    <main>
      <h1 className="text-xl font-semibold mb-4">Analytics</h1>
      <label>Platform: <select className="border p-1" value={platform} onChange={(e) => setPlatform(e.target.value)}>
        <option value="youtube">YouTube</option>
        <option value="instagram">Instagram</option>
        <option value="tiktok">TikTok</option>
        <option value="facebook">Facebook</option>
      </select></label>
      <Button onClick={collect} className="ml-2">Collect now</Button>
      <Card className="mt-6">
        <Line data={chartData} />
      </Card>
      {isLoading && <div className="text-gray-500 mt-2 flex items-center gap-2"><Spinner /> <span>Loading...</span></div>}
      <pre className="mt-4 bg-gray-900 text-green-400 p-3 rounded max-w-3xl overflow-auto">{JSON.stringify(data, null, 2)}</pre>
    </main>
  );
}
