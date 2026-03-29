# Web App — GCR Backend Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the Next.js web app so it can switch between a local FastAPI server and Google Cloud Run backend via a visible UI toggle, while keeping the existing Supabase/callback job flow intact.

**Architecture:** A `BackendConfig` module centralizes backend URL resolution. A React Context holds the active choice. A `BackendToggle` component lets the user switch at runtime (persisted in localStorage). All existing API routes (`/api/inference/*`) read the resolved URL from the module instead of directly from `process.env.INFERENCE_API_URL`.

**Tech Stack:** Next.js (App Router), React Context + localStorage, TypeScript, Vercel environment variables, shadcn/ui (for toggle component styling)

---

## Role & Context for the Executing Agent

You are updating a **Next.js + Supabase** web app that calls a Python FastAPI inference server for AI squat analysis. The Python server has just been deployed to **Google Cloud Run (GCR)**, and you need to:

> **Execution boundary:** Implement these changes in the **Next.js web app repository**. This file lives in the AI backend repo as a handoff plan/reference document.

1. Add a UI toggle so the user can switch between the local server (`http://localhost:8000`) and the GCR cloud endpoint
2. Persist the toggle choice in `localStorage` so it survives page refreshes
3. Keep every existing API route and Supabase flow **working exactly as before** — you are only changing *which backend URL* is used

**Do NOT modify:**
- The Python inference server code
- Supabase schema or RLS policies
- The callback flow (`POST /api/inference/callback`)
- The upload or polling logic

**Backend URL slots (you will introduce two):**

| Slot | Env var | Example value | When used |
|------|---------|---------------|-----------|
| Local | `NEXT_PUBLIC_LOCAL_BACKEND_URL` | `http://localhost:8000` | Dev default |
| GCR | `NEXT_PUBLIC_GCR_BACKEND_URL` | `https://exevision-api-xxxx-as.a.run.app` | Cloud target |

> **Note:** The existing `INFERENCE_API_URL` env var is server-side only (used in API routes). You will keep it but supplement it with the above two `NEXT_PUBLIC_*` vars which are readable by the browser so the toggle can work client-side.

**GCR cold start warning:** The GCR server scales to zero when idle. First request after idle can take 30–60 seconds. Your UI must handle this: show a "Warming up server..." state after 5 seconds and retry health checks with backoff for up to 90 seconds before declaring the backend unavailable.

**Current file structure to extend:**
```
app/
└── api/
    └── inference/
        ├── analyze/route.ts       ← reads INFERENCE_API_URL (keep, supplement)
        ├── submit/route.ts        ← reads INFERENCE_API_URL (keep, supplement)
        ├── status/[jobId]/route.ts
        └── callback/route.ts
lib/
└── inference-client.ts            ← typed wrapper for API calls
types/
└── analysis.ts
components/
└── analysis/
    └── ... (existing components)
```

---

## File Map (files to create or modify)

| File | Action | Purpose |
|------|--------|---------|
| `lib/backendConfig.ts` | **Create** | Single source of truth for backend URL resolution |
| `contexts/BackendContext.tsx` | **Create** | React Context — holds active backend choice + setter |
| `components/BackendToggle.tsx` | **Create** | UI toggle button (local ↔ GCR), reads/writes Context |
| `app/api/inference/analyze/route.ts` | **Modify** | Read backend URL from request header instead of only env var |
| `app/api/inference/submit/route.ts` | **Modify** | Same — use resolved backend URL |
| `app/api/inference/health/route.ts` | **Create** | Proxy `/health` to selected backend; used by toggle to verify connectivity |
| `app/layout.tsx` | **Modify** | Wrap with `BackendProvider` and render `<BackendToggle />` |
| `.env.local` | **Modify** | Add two new `NEXT_PUBLIC_*` vars |
| `apps/web/INTEGRATION_PROMPT.md` (in the AI repo) | No change | Reference only — do not edit |

---

## Task 1: Create `lib/backendConfig.ts`

**Files:**
- Create: `lib/backendConfig.ts`

This is the single source of truth for URL resolution. It works in both server-side (API routes) and client-side contexts.

- [ ] **Step 1.1: Create `lib/backendConfig.ts`**

```typescript
// lib/backendConfig.ts
// Resolves which backend URL to use: local FastAPI or Google Cloud Run.
//
// Priority order (highest wins):
//   1. Value stored in localStorage (user's runtime toggle choice)
//   2. NEXT_PUBLIC_BACKEND_OVERRIDE env var (per-deployment override)
//   3. Default: LOCAL in dev, GCR in production
//
// Keep `INFERENCE_API_URL` as server-side fallback for API routes.

export type BackendTarget = 'local' | 'gcr';

const VALID_TARGETS: BackendTarget[] = ['local', 'gcr'];

function normalizeTarget(value?: string | null): BackendTarget | null {
  if (!value) return null;
  return VALID_TARGETS.includes(value as BackendTarget)
    ? (value as BackendTarget)
    : null;
}

export const BACKEND_URLS: Record<BackendTarget, string> = {
  local: process.env.NEXT_PUBLIC_LOCAL_BACKEND_URL ?? 'http://localhost:8000',
  gcr: process.env.NEXT_PUBLIC_GCR_BACKEND_URL ?? '',
};

export const STORAGE_KEY = 'exevision_backend_target';

/** Default target based on environment. */
export function getDefaultTarget(): BackendTarget {
  if (typeof window === 'undefined') {
    // Server-side: use validated env override if set
    const override = normalizeTarget(process.env.NEXT_PUBLIC_BACKEND_OVERRIDE);
    return override ?? (process.env.NODE_ENV === 'production' ? 'gcr' : 'local');
  }
  // Client-side default (before localStorage is read)
  return process.env.NODE_ENV === 'production' ? 'gcr' : 'local';
}

/** Read the user's persisted toggle choice from localStorage. */
export function getStoredTarget(): BackendTarget | null {
  if (typeof window === 'undefined') return null;
  const val = localStorage.getItem(STORAGE_KEY);
  return normalizeTarget(val);
}

/** Persist toggle choice to localStorage. */
export function setStoredTarget(target: BackendTarget): void {
  if (typeof window === 'undefined') return;
  localStorage.setItem(STORAGE_KEY, target);
}

/**
 * Resolve the active backend URL.
 * On the client: prefers localStorage, then env default.
 * On the server (API routes): pass X-Backend-Target and INFERENCE_API_URL fallback.
 */
export function resolveBackendURL(
  headerTarget?: string | null,
  serverFallbackURL?: string
): string {
  const target = normalizeTarget(headerTarget) ?? getStoredTarget() ?? getDefaultTarget();

  const url = BACKEND_URLS[target];
  if (url) return url;

  if (serverFallbackURL) return serverFallbackURL;

  console.warn(`[backendConfig] No URL configured for target "${target}", falling back to local`);
  return BACKEND_URLS.local;
}
```

- [ ] **Step 1.2: Verify TypeScript compiles**

```bash
npx tsc --noEmit
```

Expected: no errors in `lib/backendConfig.ts`

- [ ] **Step 1.3: Commit**

```bash
git add lib/backendConfig.ts
git commit -m "feat: add backendConfig module for local/GCR URL resolution"
```

---

## Task 2: Create `contexts/BackendContext.tsx`

**Files:**
- Create: `contexts/BackendContext.tsx`

React Context that holds the active target and provides a setter. Components use `useBackend()` to read and update the selection.

- [ ] **Step 2.1: Create `contexts/BackendContext.tsx`**

```tsx
'use client';

// contexts/BackendContext.tsx
import React, { createContext, useContext, useEffect, useState } from 'react';
import {
  BackendTarget,
  BACKEND_URLS,
  getDefaultTarget,
  getStoredTarget,
  setStoredTarget,
} from '@/lib/backendConfig';

interface BackendContextValue {
  target: BackendTarget;
  backendURL: string;
  setTarget: (t: BackendTarget) => void;
}

const BackendContext = createContext<BackendContextValue | null>(null);

export function BackendProvider({ children }: { children: React.ReactNode }) {
  const [target, setTargetState] = useState<BackendTarget>(getDefaultTarget);

  // Hydrate from localStorage after mount (avoids SSR mismatch)
  useEffect(() => {
    const stored = getStoredTarget();
    if (stored) setTargetState(stored);
  }, []);

  const setTarget = (t: BackendTarget) => {
    setTargetState(t);
    setStoredTarget(t);
  };

  return (
    <BackendContext.Provider
      value={{ target, backendURL: BACKEND_URLS[target], setTarget }}
    >
      {children}
    </BackendContext.Provider>
  );
}

export function useBackend(): BackendContextValue {
  const ctx = useContext(BackendContext);
  if (!ctx) throw new Error('useBackend must be used inside <BackendProvider>');
  return ctx;
}
```

- [ ] **Step 2.2: Verify TypeScript compiles**

```bash
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 2.3: Commit**

```bash
git add contexts/BackendContext.tsx
git commit -m "feat: add BackendContext for runtime backend target state"
```

---

## Task 3: Create `components/BackendToggle.tsx`

**Files:**
- Create: `components/BackendToggle.tsx`

Visible toggle button that shows current backend target and lets the user switch. Calls `/api/inference/health` to show live connectivity status for the selected backend.

- [ ] **Step 3.1: Create `components/BackendToggle.tsx`**

```tsx
'use client';

// components/BackendToggle.tsx
// Default policy: hide in production unless NEXT_PUBLIC_SHOW_BACKEND_TOGGLE === 'true'.
import { useEffect, useState } from 'react';
import { useBackend } from '@/contexts/BackendContext';
import { BackendTarget } from '@/lib/backendConfig';

type HealthStatus = 'unknown' | 'checking' | 'ok' | 'error';

export function BackendToggle() {
  const { target, setTarget } = useBackend();
  const [health, setHealth] = useState<HealthStatus>('unknown');

  // Check health whenever target changes
  useEffect(() => {
    setHealth('checking');
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 10_000);

    fetch(`/api/inference/health?target=${target}`, { signal: controller.signal })
      .then((r) => setHealth(r.ok ? 'ok' : 'error'))
      .catch(() => setHealth('error'))
      .finally(() => clearTimeout(timer));

    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [target]);

  const toggle = () => setTarget(target === 'local' ? 'gcr' : 'local');

  const statusColor: Record<HealthStatus, string> = {
    unknown: 'bg-gray-400',
    checking: 'bg-yellow-400 animate-pulse',
    ok: 'bg-green-500',
    error: 'bg-red-500',
  };

  const statusLabel: Record<HealthStatus, string> = {
    unknown: '',
    checking: 'checking…',
    ok: 'online',
    error: 'offline',
  };

  return (
    <div className="fixed bottom-4 right-4 z-50 flex items-center gap-2 rounded-full border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-xs text-zinc-200 shadow-lg">
      {/* Health indicator dot */}
      <span className={`h-2 w-2 rounded-full ${statusColor[health]}`} />

      {/* Current backend label */}
      <span className="font-mono">
        {target === 'local' ? 'local :8000' : 'gcr'}
      </span>

      {/* Status text */}
      {health !== 'unknown' && (
        <span className="text-zinc-500">{statusLabel[health]}</span>
      )}

      {/* Toggle button */}
      <button
        onClick={toggle}
        className="ml-1 rounded px-2 py-0.5 text-xs bg-zinc-700 hover:bg-zinc-600 transition-colors"
        title={`Switch to ${target === 'local' ? 'GCR' : 'local'} backend`}
      >
        switch
      </button>
    </div>
  );
}
```

> **Production safety:** keep this toggle hidden in production unless explicitly enabled with `NEXT_PUBLIC_SHOW_BACKEND_TOGGLE=true`.

- [ ] **Step 3.2: Commit**

```bash
git add components/BackendToggle.tsx
git commit -m "feat: add BackendToggle component with live health indicator"
```

---

## Task 4: Create `/api/inference/health` Proxy Route

**Files:**
- Create: `app/api/inference/health/route.ts`

Proxies the `/health` endpoint to whichever backend is selected. Called by `BackendToggle` with `?target=local|gcr`.

- [ ] **Step 4.1: Create `app/api/inference/health/route.ts`**

```typescript
// app/api/inference/health/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { resolveBackendURL } from '@/lib/backendConfig';

export async function GET(req: NextRequest) {
  const target = req.nextUrl.searchParams.get('target');
  const backendURL = resolveBackendURL(target, process.env.INFERENCE_API_URL);

  const maxWarmupMs = target === 'gcr' ? 90_000 : 10_000;
  const startedAt = Date.now();
  let attempt = 0;
  let lastError: unknown = null;

  while (Date.now() - startedAt < maxWarmupMs) {
    attempt += 1;
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 12_000);

      const upstream = await fetch(`${backendURL}/health`, {
        signal: controller.signal,
        headers: { Authorization: `Bearer ${process.env.INFERENCE_API_SECRET ?? ''}` },
      });

      clearTimeout(timeout);

      // Return immediately on any successful response from backend.
      const body = await upstream.json();
      return NextResponse.json(body, { status: upstream.status });
    } catch (err) {
      lastError = err;
      const elapsed = Date.now() - startedAt;
      const remaining = maxWarmupMs - elapsed;
      if (remaining <= 0) break;

      const backoffMs = Math.min(500 * attempt, 3_000, remaining);
      await new Promise((resolve) => setTimeout(resolve, backoffMs));
    }
  }

  return NextResponse.json(
    {
      status: 'error',
      reason: 'timeout',
      hint: 'Backend unreachable after warmup retries (up to 90s). GCR may still be cold-starting.',
      detail: String(lastError ?? 'unknown error'),
    },
    { status: 503 }
  );
}
```

- [ ] **Step 4.2: Test the health route**

Start the local Python server, then in a browser or terminal:
```bash
curl "http://localhost:3000/api/inference/health?target=local"
```

Expected:
```json
{"status": "ok", "stages_dir_ok": true, "models_ok": true}
```

With the server stopped:
```bash
curl "http://localhost:3000/api/inference/health?target=local"
```

Expected:
```json
{"status": "error", "reason": "timeout", "hint": "..."}
```

- [ ] **Step 4.3: Commit**

```bash
git add app/api/inference/health/route.ts
git commit -m "feat: add /api/inference/health proxy route for backend toggle"
```

---

## Task 5: Update Existing API Routes to Use `resolveBackendURL`

**Files:**
- Modify: `app/api/inference/analyze/route.ts`
- Modify: `app/api/inference/submit/route.ts`

Currently these routes read `process.env.INFERENCE_API_URL` directly. Update them to use `resolveBackendURL()` with the `X-Backend-Target` header so the frontend toggle propagates to server-side calls.

**How the header propagates:**
Client → calls Next.js API route with `X-Backend-Target: local|gcr` → API route calls `resolveBackendURL(header)` → calls correct backend

- [ ] **Step 5.1: Update `app/api/inference/analyze/route.ts`**

Find the line that reads `process.env.INFERENCE_API_URL` (likely used in a fetch call). Replace it:

```typescript
// BEFORE (find this pattern in the file):
const inferenceURL = process.env.INFERENCE_API_URL;

// AFTER (replace with):
import { resolveBackendURL } from '@/lib/backendConfig';
// ...inside the handler:
const backendTarget = req.headers.get('X-Backend-Target');
const inferenceURL = resolveBackendURL(backendTarget, process.env.INFERENCE_API_URL);
```

Full pattern: add the import at the top and replace direct `process.env.INFERENCE_API_URL` usage with `resolveBackendURL(req.headers.get('X-Backend-Target'), process.env.INFERENCE_API_URL)`.

> **Do not change** the Supabase calls, the auth check, or the callback URL logic. Only the `inferenceURL` assignment changes.

- [ ] **Step 5.2: Update `app/api/inference/submit/route.ts`**

Apply the same change as Step 5.1:
- Add import: `import { resolveBackendURL } from '@/lib/backendConfig';`
- Replace `process.env.INFERENCE_API_URL` with `resolveBackendURL(req.headers.get('X-Backend-Target'), process.env.INFERENCE_API_URL)`

- [ ] **Step 5.3: Verify TypeScript compiles**

```bash
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 5.4: Commit**

```bash
git add app/api/inference/analyze/route.ts app/api/inference/submit/route.ts
git commit -m "feat: update inference API routes to use resolveBackendURL"
```

---

## Task 6: Update Client-Side API Calls to Pass `X-Backend-Target` Header

**Files:**
- Modify: `lib/inference-client.ts` (or wherever client-side `fetch` calls to `/api/inference/*` are made)

The client needs to pass `X-Backend-Target: local|gcr` on every call so the server routes know which backend to forward to.

- [ ] **Step 6.1: Update `lib/inference-client.ts`**

Find the fetch wrapper function(s) in this file. Add the `X-Backend-Target` header. Example:

```typescript
// lib/inference-client.ts (add this helper at the top)
import { getStoredTarget, getDefaultTarget } from '@/lib/backendConfig';

function getBackendTargetHeader(): string {
  return getStoredTarget() ?? getDefaultTarget();
}

// In every fetch call that goes to /api/inference/*, add the header:
// BEFORE:
const response = await fetch('/api/inference/analyze', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(payload),
});

// AFTER:
const response = await fetch('/api/inference/analyze', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-Backend-Target': getBackendTargetHeader(),
  },
  body: JSON.stringify(payload),
});
```

Apply this pattern to **every** fetch call in this file that calls `/api/inference/analyze` or `/api/inference/submit`. Do NOT modify the `/api/inference/callback` call (that's server-to-server).

- [ ] **Step 6.2: Verify TypeScript compiles**

```bash
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 6.3: Commit**

```bash
git add lib/inference-client.ts
git commit -m "feat: pass X-Backend-Target header on inference API calls"
```

---

## Task 7: Wire `BackendProvider` and `BackendToggle` into Layout

**Files:**
- Modify: `app/layout.tsx`

- [ ] **Step 7.1: Update `app/layout.tsx`**

Add the provider and toggle. Find the root layout component and make these two changes:

```tsx
// app/layout.tsx
// 1. Add imports at the top:
import { BackendProvider } from '@/contexts/BackendContext';
import { BackendToggle } from '@/components/BackendToggle';

// 2. Wrap children with BackendProvider and add BackendToggle inside the body:
// BEFORE:
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

// AFTER:
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <BackendProvider>
          {children}
          {process.env.NEXT_PUBLIC_SHOW_BACKEND_TOGGLE === 'true' && <BackendToggle />}
          {/* Recommended: set NEXT_PUBLIC_SHOW_BACKEND_TOGGLE=true only in dev/preview */}
        </BackendProvider>
      </body>
    </html>
  );
}
```

- [ ] **Step 7.2: Start the dev server and verify the toggle appears**

```bash
npm run dev
```

Open `http://localhost:3000`. Expected: a small floating pill in the bottom-right corner showing `local :8000` with a health indicator dot.

- [ ] **Step 7.3: Test the toggle**

1. Click "switch" — label changes to `gcr`
2. Health dot turns yellow (checking) then red (no GCR URL configured yet — expected)
3. Refresh the page — `gcr` choice persists (localStorage)
4. Click "switch" again — returns to `local`, dot turns green (if Python server is running)

- [ ] **Step 7.4: Commit**

```bash
git add app/layout.tsx
git commit -m "feat: wire BackendProvider and BackendToggle into root layout"
```

---

## Task 8: Update Environment Variables

**Files:**
- Modify: `.env.local`
- Update Vercel environment variables (via dashboard or `vercel env` CLI)

- [ ] **Step 8.1: Update `.env.local`**

Add these lines to your `.env.local` (existing vars are unchanged):

```env
# Backend switching — local dev server
NEXT_PUBLIC_LOCAL_BACKEND_URL=http://localhost:8000

# Backend switching — Google Cloud Run
# Replace with your actual GCR service URL from the deployment plan
NEXT_PUBLIC_GCR_BACKEND_URL=https://exevision-api-xxxx-as.a.run.app

# Toggle visibility (recommended: true only in dev/preview)
NEXT_PUBLIC_SHOW_BACKEND_TOGGLE=true

# Keep the existing server-side var as the default for production API routes
# This is still used as the fallback when no X-Backend-Target header is present
INFERENCE_API_URL=https://exevision-api-xxxx-as.a.run.app
INFERENCE_API_SECRET=your-shared-secret-here
```

- [ ] **Step 8.2: Set Vercel environment variables for production**

Using the Vercel dashboard or CLI, add:

| Variable | Value | Environment |
|----------|-------|-------------|
| `NEXT_PUBLIC_LOCAL_BACKEND_URL` | `http://localhost:8000` | Development |
| `NEXT_PUBLIC_GCR_BACKEND_URL` | `https://exevision-api-xxxx-as.a.run.app` | All |
| `NEXT_PUBLIC_SHOW_BACKEND_TOGGLE` | `true` | Development, Preview |
| `INFERENCE_API_URL` | `https://exevision-api-xxxx-as.a.run.app` | Production, Preview |
| `INFERENCE_API_SECRET` | (your secret) | All |

CLI equivalent:
```bash
vercel env add NEXT_PUBLIC_GCR_BACKEND_URL production
# → paste: https://exevision-api-xxxx-as.a.run.app

vercel env add INFERENCE_API_URL production
# → paste: https://exevision-api-xxxx-as.a.run.app
```

- [ ] **Step 8.3: Add `.env.local` to `.gitignore` if not already there**

```bash
grep ".env.local" .gitignore || echo ".env.local" >> .gitignore
```

- [ ] **Step 8.4: Commit the gitignore update (not the .env.local itself)**

```bash
git add .gitignore
git commit -m "chore: ensure .env.local is gitignored"
```

---

## Task 9: Handle GCR Cold Start in the UI

**Files:**
- Modify: wherever the `/analyze` page calls health check or initiates a job (likely `app/analyze/page.tsx`)

GCR scales to zero when idle. The first request after idle takes 30–60s. The user needs feedback.

- [ ] **Step 9.1: Add cold-start awareness to the analyze page**

Find the section in `app/analyze/page.tsx` (or equivalent) that calls the health check or submits the job. Add a warming state:

```tsx
// In your analyze submission handler:
const [isWarming, setIsWarming] = useState(false);

async function waitForBackendReady(target: 'local' | 'gcr', maxWaitMs?: number) {
  const effectiveMaxWaitMs = maxWaitMs ?? (target === 'gcr' ? 90_000 : 10_000);
  const startedAt = Date.now();
  let attempt = 0;

  while (Date.now() - startedAt < effectiveMaxWaitMs) {
    attempt += 1;
    try {
      const response = await fetch(`/api/inference/health?target=${target}`);
      const health = await response.json();
      if (response.ok && health.status === 'ok') return true;
    } catch {
      // Keep retrying until timeout.
    }

    const elapsed = Date.now() - startedAt;
    const remaining = effectiveMaxWaitMs - elapsed;
    if (remaining <= 0) break;

    const backoffMs = Math.min(500 * attempt, 3_000, remaining);
    await new Promise((resolve) => setTimeout(resolve, backoffMs));
  }

  return false;
}

async function handleAnalyze() {
  // 1. Check health first; if it takes >5s, show warming state.
  const warmingTimer = setTimeout(() => setIsWarming(true), 5_000);

  try {
    const ready = await waitForBackendReady(target, 90_000);
    clearTimeout(warmingTimer);
    setIsWarming(false);

    if (!ready) {
      // Show error: backend not ready after warmup retries.
      return;
    }
    // Proceed with job submission...
  } catch {
    clearTimeout(warmingTimer);
    setIsWarming(false);
    // Show error
  }
}

// In JSX, show warming message when isWarming is true:
{isWarming && (
  <div className="rounded-md bg-yellow-950 border border-yellow-700 p-3 text-sm text-yellow-300">
    Warming up the analysis server (GCR cold start)... this can take up to 60s.
  </div>
)}
```

- [ ] **Step 9.2: Commit**

```bash
git add app/analyze/page.tsx  # adjust path to match your file
git commit -m "feat: show GCR cold-start warming message during health check"
```

---

## Task 10: End-to-End Integration Test

- [ ] **Step 10.1: Test with local backend**

1. Start the Python server:
   ```bash
   # In the exevision_modelAI repo:
   INFERENCE_API_SECRET=dev-secret uvicorn apps.api.main:app --port 8000
   ```
2. Start Next.js:
   ```bash
   npm run dev
   ```
3. Open `http://localhost:3000`
4. Verify toggle shows `local :8000` with green dot
5. Upload a short squat video → verify job queues and completes

- [ ] **Step 10.2: Test with GCR backend via toggle**

1. Click "switch" in the toggle → should show `gcr` with green dot (if GCR is deployed)
2. Upload the same video → verify job routes to GCR
3. Check that the GCR Cloud Run logs show the job (optional but useful):
   ```bash
   gcloud logging read "resource.type=cloud_run_revision" --limit=20
   ```

- [ ] **Step 10.3: Test cold start UX**

1. Wait 5+ minutes for GCR to scale to zero (or trigger manually from GCP console)
2. Submit a job — verify the "Warming up…" message appears within 5s
3. After the server warms (~30-60s), verify the job proceeds normally

- [ ] **Step 10.4: Test toggle persistence**

1. Set toggle to `gcr`
2. Refresh page
3. Verify toggle still shows `gcr` (localStorage persisted)

- [ ] **Step 10.5: Test fallback behavior when target header is missing/invalid**

1. Trigger an analyze or submit request without `X-Backend-Target`
2. Verify server route falls back to `INFERENCE_API_URL`
3. Trigger a request with `X-Backend-Target: invalid-value`
4. Verify invalid target is ignored and `INFERENCE_API_URL` fallback is used

---

## Quick Reference: How to Enable the Toggle in Production (If Needed)

Default recommendation: keep the toggle disabled in production. Only enable temporarily for diagnostics.

In `app/layout.tsx`, use:

```tsx
// Recommended production-safe pattern:
{process.env.NEXT_PUBLIC_SHOW_BACKEND_TOGGLE === 'true' && <BackendToggle />}
```

Set `NEXT_PUBLIC_SHOW_BACKEND_TOGGLE=true` only in environments where switching is intended (usually Development/Preview).

---

## Critical Reminders

1. **The `callback_url` in `POST /infer` must always be your public Vercel URL**, not the local URL. The Python server (running on GCR or locally) needs to reach it externally. In dev, use something like [ngrok](https://ngrok.com) or just omit `callback_url` and rely on polling.
2. **`INFERENCE_API_SECRET` must match on both sides** — same value in `.env.local` and in GCR env vars.
3. **Never log `INFERENCE_API_SECRET`** — it's a Bearer token.
4. **Signed video URLs must have ≥ 300s expiry** — GCR pipeline can take 2+ minutes.
5. **`X-Backend-Target` header only affects Next.js proxy routes**, not the callback. The callback comes from the Python server back to Next.js — it always uses your public Vercel URL regardless of toggle.
