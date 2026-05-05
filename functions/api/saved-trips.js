/* GET/POST/DELETE /api/saved-trips — Saved trips feature for the dashboard.
   Stores trip snapshots per-user (JWT sub or 'anonymous'). The UI ALSO
   mirrors saved trips to localStorage for offline + cross-isolate use. */

import { jsonResponse, verifyJwt, getJwtSecret } from "./_shared/auth.js";

const TRIPS = globalThis.__SR_SAVED_TRIPS__ ||= new Map();

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
  const trips = TRIPS.get(uid) || [];
  return jsonResponse({
    ok: true,
    count: trips.length,
    trips: [...trips].sort((a, b) => b.savedAt - a.savedAt),
  });
};

export const onRequestPost = async ({ request }) => {
  let body = {};
  try { body = await request.json(); } catch {}
  const { destination, origin, days, budget, persona, summary, heroImage, totalAttractions } = body || {};
  if (!destination) return jsonResponse({ ok: false, error: "destination required" }, 400);
  const uid = await getUserId(request);
  const trip = {
    id: Date.now(),
    destination, origin: origin || null,
    days: Number(days) || 3,
    budget: Number(budget) || 15000,
    persona: persona || "explorer",
    summary: summary || `${days || 3}-day trip to ${destination}`,
    heroImage: heroImage || null,
    totalAttractions: Number(totalAttractions) || 0,
    savedAt: Date.now(),
    savedDate: new Date().toLocaleDateString("en-IN"),
  };
  const list = TRIPS.get(uid) || [];
  // Dedupe by destination+days+persona
  const filtered = list.filter(t => !(t.destination === destination && t.days === trip.days && t.persona === trip.persona));
  filtered.push(trip);
  if (filtered.length > 50) filtered.shift();
  TRIPS.set(uid, filtered);
  return jsonResponse({ ok: true, trip, totalTrips: filtered.length });
};

export const onRequestDelete = async ({ request }) => {
  const url = new URL(request.url);
  const id = parseInt(url.searchParams.get("id") || "0");
  if (!id) return jsonResponse({ ok: false, error: "id required" }, 400);
  const uid = await getUserId(request);
  const list = TRIPS.get(uid) || [];
  const next = list.filter(t => t.id !== id);
  TRIPS.set(uid, next);
  return jsonResponse({ ok: true, removed: list.length - next.length, totalTrips: next.length });
};
