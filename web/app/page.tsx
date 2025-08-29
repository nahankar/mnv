"use client";

import Link from 'next/link';
import useSWR from 'swr';
import { Card } from '@/components/ui/card';
import { Spinner } from '@/components/ui';

export default function Page() {
  const { data, isLoading } = useSWR('/api/bff/health', (url) => fetch(url).then(r => r.json()))
  const list = data?.services || [];
  return (
    <main>
      <h1 className="text-xl font-semibold mb-2">AI Pipeline Admin</h1>
      <div className="flex gap-3 flex-wrap">
        <Link href="/triggers">Triggers</Link>
        <Link href="/review-queue">Review Queue</Link>
        <Link href="/analytics">Analytics</Link>
      </div>
      <h2 className="mt-6 font-medium">Service Health</h2>
      {isLoading ? <div className="text-gray-500 flex items-center gap-2"><Spinner /> <span>Loading...</span></div> : (
        <Card>
          <ul>
            {list.map((x: any) => (
              <li key={x.name}>{x.name}: {x.status}</li>
            ))}
          </ul>
        </Card>
      )}
    </main>
  );
}
