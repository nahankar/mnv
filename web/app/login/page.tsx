"use client";

import { signIn } from "next-auth/react";
import { useState } from "react";

export default function LoginPage() {
  const [err, setErr] = useState<string | null>(null);
  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const fd = new FormData(e.currentTarget);
    const username = fd.get("username") as string;
    const password = fd.get("password") as string;
    const res = await signIn("credentials", {
      username,
      password,
      redirect: true,
      callbackUrl: "/"
    });
    if (res?.error) setErr(res.error);
  }
  return (
    <main style={{ padding: 24 }}>
      <h1>Login</h1>
      <form onSubmit={onSubmit} style={{ display: 'grid', gap: 8, maxWidth: 320 }}>
        <label>Username<input name="username" defaultValue="admin" /></label>
        <label>Password<input type="password" name="password" defaultValue="admin" /></label>
        <button type="submit">Sign In</button>
      </form>
      {err && <p style={{ color: 'red' }}>{err}</p>}
    </main>
  );
}
