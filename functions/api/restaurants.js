/* POST /api/restaurants — curated real restaurants per city. */

import { jsonResponse } from "./_shared/auth.js";
import { generateRestaurants } from "./_shared/extras.js";
import { geocode } from "./_shared/cities.js";

export const onRequestOptions = () => jsonResponse({ ok: true });

export const onRequestPost = async ({ request }) => {
  let body = {};
  try { body = await request.json(); } catch { body = {}; }
  const { destination, city } = body || {};
  const place = destination || city;
  if (!place) return jsonResponse({ ok: false, error: "Destination required." }, 400);
  const g = geocode(place);
  const restaurants = generateRestaurants(place, g.lat, g.lon);
  return jsonResponse({ ok: true, restaurants });
};
