/* GET/POST /api/journal/entries — Trip journal feature ported from the
   original GitHub backend (NOMAD notes concept). Persists entries in an
   in-isolate Map keyed by user (Bearer JWT sub or 'anonymous'). The UI
   ALSO mirrors entries to localStorage for offline + cross-isolate use. */

import { jsonResponse, verifyJwt, getJwtSecret } from "../_shared/auth.js";

const JOURNAL = globalThis.__SR_JOURNAL__ ||= new Map();

async function getUserId(request) {
  const auth = request.headers.get("Authorization") || "";
  const m = auth.match(/^Bearer\s+(.+)$/i);
  if (!m) return "anonymous";
  try {
    const claims = await verifyJwt(m[1], getJwtSecret());
    return claims?.sub || claims?.email || "anonymous";
  } catch { return "anonymous"; }
}

export const onRequestOptions = () => jsonResponse({ ok: true });

export const onRequestGet = async ({ request }) => {
  const uid = await getUserId(request);
  const entries = JOURNAL.get(uid) || [];
  return jsonResponse({
    ok: true,
    count: entries.length,
    entries: [...entries].sort((a, b) => b.id - a.id),
  });
};

export const onRequestPost = async ({ request }) => {
  let body = {};
  try { body = await request.json(); } catch {}
  const { text, destination, mood, photoUrl } = body || {};
  if (!text || !String(text).trim()) {
    return jsonResponse({ ok: false, error: "Entry text required" }, 400);
  }
  const uid = await getUserId(request);
  const entry = {
    id: Date.now(),
    text: String(text).slice(0, 2000).trim(),
    destination: destination || "General",
    mood: mood || null,
    photoUrl: photoUrl || null,
    date: new Date().toISOString(),
    dateLabel: new Date().toLocaleString(),
  };
  const list = JOURNAL.get(uid) || [];
  list.push(entry);
  if (list.length > 200) list.shift();
  JOURNAL.set(uid, list);
  return jsonResponse({ ok: true, entry, totalEntries: list.length });
};

export const onRequestDelete = async ({ request }) => {
  const url = new URL(request.url);
  const id = parseInt(url.searchParams.get("id") || "0");
  if (!id) return jsonResponse({ ok: false, error: "id required" }, 400);
  const uid = await getUserId(request);
  const list = JOURNAL.get(uid) || [];
  const next = list.filter(e => e.id !== id);
  JOURNAL.set(uid, next);
  return jsonResponse({ ok: true, removed: list.length - next.length, totalEntries: next.length });
};
