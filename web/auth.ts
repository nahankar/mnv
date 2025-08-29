import NextAuth from "next-auth";
import Credentials from "next-auth/providers/credentials";
import { z } from "zod";

const schema = z.object({ username: z.string().min(1), password: z.string().min(1) });

export const { handlers: { GET, POST }, auth, signIn, signOut } = NextAuth({
  session: { strategy: "jwt" },
  callbacks: {
    async jwt({ token, user }) {
      if (user) {
        // RBAC mapping
        let role = (user as any).role || 'viewer';
        if (!role && (user as any).id?.toString()?.startsWith('op_')) role = 'operator';
        (token as any).role = role;
      }
      return token;
    },
    async session({ session, token }) {
      (session as any).user = { ...(session as any).user, role: (token as any).role || 'viewer' };
      return session;
    }
  },
  providers: [
    Credentials({
      name: "Credentials",
      credentials: {
        username: { label: "Username", type: "text" },
        password: { label: "Password", type: "password" }
      },
      authorize: async (credentials) => {
        const parsed = schema.safeParse(credentials);
        if (!parsed.success) return null;
        const { username, password } = parsed.data;
        const adminUser = process.env.ADMIN_USER || "admin";
        const adminPass = process.env.ADMIN_PASS || "admin";
        if (username === adminUser && password === adminPass) {
          return { id: "admin", name: "Admin", role: "admin" } as any;
        }
        // fallback viewer user for demo; in prod, remove this
        if (username && password) {
          return { id: username, name: username, role: "viewer" } as any;
        }
        return null;
      }
    })
  ],
  cookies: {
    sessionToken: {
      name: "__Host-next-auth.session-token",
      options: { httpOnly: true, sameSite: "lax", path: "/", secure: process.env.NODE_ENV === "production" }
    }
  }
});
