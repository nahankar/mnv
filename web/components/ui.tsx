"use client";

import clsx from 'clsx';
import toast from 'react-hot-toast';

export function Button(props: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  const { className, disabled, ...rest } = props;
  return <button {...rest} disabled={disabled} className={clsx('px-3 py-1 rounded bg-blue-600 text-white disabled:opacity-50', className)} />
}

export function DangerButton(props: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  const { className, disabled, ...rest } = props;
  return <button {...rest} disabled={disabled} className={clsx('px-3 py-1 rounded bg-red-600 text-white disabled:opacity-50', className)} />
}

export function Card({ children, className }: { children: React.ReactNode, className?: string }) {
  return <div className={clsx("bg-white p-4 rounded shadow", className)}>{children}</div>
}

export function Spinner() {
  return <div className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-current border-r-transparent align-[-0.125em] text-blue-600 motion-reduce:animate-[spin_1.5s_linear_infinite]" role="status" aria-label="loading" />
}

export const notify = {
  success: (m: string) => toast.success(m),
  error: (m: string) => toast.error(m)
}
