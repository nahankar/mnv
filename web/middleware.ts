import { auth } from "./auth";
import { NextResponse, type NextRequest } from "next/server";

export async function middleware(req: NextRequest) {
  const session = await auth();
  const isAuthRoute = req.nextUrl.pathname.startsWith("/login") || req.nextUrl.pathname.startsWith("/api/auth");
  if (!session && !isAuthRoute) {
    const url = req.nextUrl.clone();
    url.pathname = "/login";
    return NextResponse.redirect(url);
  }
  // RBAC route guards
  const role = (session as any)?.user?.role || 'viewer';
  const adminOnly = ["/model-configs", "/budgets", "/api/bff/budgets", "/api/bff/model-configs"];
  const operatorAllowed = ["/review-queue", "/api/bff/review"]; // viewer can't access
  const path = req.nextUrl.pathname;
  if (adminOnly.some(p => path.startsWith(p)) && role !== 'admin') {
    return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
  }
  if (operatorAllowed.some(p => path.startsWith(p)) && role === 'viewer') {
    return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
  }
  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
