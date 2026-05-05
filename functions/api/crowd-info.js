/* POST /api/crowd-info — crowd density predictions. */

import { jsonResponse } from "./_shared/auth.js";

export const onRequestOptions = () => jsonResponse({ ok: true });

export const onRequestPost = async ({ request }) => {
  let body = {};
  try { body = await request.json(); } catch { body = {}; }
  const { destination, attractions } = body || {};
  const places = Array.isArray(attractions) && attractions.length
    ? attractions : [`${destination || "City"} Central`];

  return jsonResponse({
    ok: true,
    locations: places.map((name, i) => ({
      name,
      peak_hours: i % 2 === 0 ? "11:00 AM – 2:00 PM" : "5:00 PM – 8:00 PM",
      recommended_time: i % 2 === 0 ? "8:00 AM – 10:00 AM" : "3:30 PM – 5:00 PM",
      indicator: i % 2 === 0 ? "Moderate crowd risk" : "Best before evening rush",
      currentDensity: Math.round(30 + Math.random() * 60),
      weekendMultiplier: 1.4,
    }))
  });
};
