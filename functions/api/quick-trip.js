/* POST /api/quick-trip — nearby short outing suggestions. */

import { jsonResponse } from "./_shared/auth.js";

export const onRequestOptions = () => jsonResponse({ ok: true });

export const onRequestPost = async ({ request }) => {
  let body = {};
  try { body = await request.json(); } catch { body = {}; }
  const { latitude, longitude, available_hours } = body || {};

  if (typeof latitude !== "number" || typeof longitude !== "number") {
    return jsonResponse({ ok: false, error: "Latitude and longitude required." }, 400);
  }
  const hours = Math.max(1, Math.min(Number(available_hours) || 4, 12));

  const places = [
    { name: "Lakeview Escape",      distance: "12 km", estimated_travel_time: "28 mins", rating: 4.6, note: `Scenic stop reachable within ${hours} hrs.` },
    { name: "Old Town Food Street", distance: "8 km",  estimated_travel_time: "22 mins", rating: 4.4, note: "Food-first short exploration." },
    { name: "Hilltop Sunset Point", distance: "18 km", estimated_travel_time: "40 mins", rating: 4.7, note: "Best photogenic views with minimal planning." },
    { name: "Heritage Bazaar",      distance: "5 km",  estimated_travel_time: "15 mins", rating: 4.3, note: "Local crafts and street food." },
  ].slice(0, Math.min(hours, 3) + 1);

  return jsonResponse({ ok: true, places });
};
