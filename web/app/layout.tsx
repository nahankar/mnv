import './globals.css'
import { Toaster } from 'react-hot-toast'
import { auth } from '@/auth'

export const metadata = { title: 'Admin Dashboard' };

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const session = await auth();
  const role = (session as any)?.user?.role || 'viewer';
  return (
    <html lang="en">
      <body className="min-h-screen bg-gray-50 text-gray-900">
        <div className="px-4 py-3 bg-gray-900 text-white flex gap-4 items-center">
          <a href="/" className="hover:underline">Dashboard</a>
          <a href="/triggers" className="hover:underline">Triggers</a>
          {role !== 'viewer' && <a href="/review-queue" className="hover:underline">Review Queue</a>}
          <a href="/analytics" className="hover:underline">Analytics</a>
          {role === 'admin' && <a href="/model-configs" className="hover:underline">Model Configs</a>}
          {role === 'admin' && <a href="/budgets" className="hover:underline">Budgets</a>}
          <a href="/runs" className="hover:underline">Runs</a>
          <span className="ml-auto"><a href="/api/auth/signout" className="hover:underline">Sign out</a></span>
        </div>
        <div className="p-6">{children}</div>
        <Toaster />
      </body>
    </html>
  );
}
