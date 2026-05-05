/* POST /api/activities/search — curated city attractions WITH coords for map. */

import { jsonResponse } from "../_shared/auth.js";
import { getTopAttractions, geocode } from "../_shared/cities.js";

export const onRequestOptions = () => jsonResponse({ ok: true });

export const onRequestPost = async ({ request }) => {
  let body = {};
  try { body = await request.json(); } catch { body = {}; }
  let { destination, latitude, longitude } = body || {};

  const place = destination || "";
  const geo   = geocode(place);
  if (!latitude || !longitude) { latitude = geo.lat; longitude = geo.lon; }

  const curated = getTopAttractions(geo.resolvedCity || place);

  if (curated.length) {
    const activities = curated.map((p, i) => ({
      id: `c${i}`,
      name: p.name,
      kinds: `${p.type}, cultural`,
      type: p.type,
      distance: `${(0.5 + i * 0.7).toFixed(1)} km`,
      rating: (4.3 + (i % 5) * 0.1).toFixed(1),
      description: p.description,
      point: { lat: p.lat, lon: p.lon },
      lat: p.lat, lon: p.lon,
      wikiTitle: p.wikiTitle,
      mapsUrl: `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(p.name + " " + place)}`,
      wikiUrl: p.wikiTitle ? `https://en.wikipedia.org/wiki/${p.wikiTitle}` : null,
    }));
    return jsonResponse({ ok: true, activities, source: "curated", center: { lat: latitude, lon: longitude } });
  }

  // Generic fallback (still with synthetic coords near destination centre)
  const fallback = ["Heritage Walk","Local Market","Scenic Viewpoint","Museum","Public Park","Temple or Shrine"]
    .map((name, i) => ({
      id: `f${i}`, name: `${place || "Local"} ${name}`,
      kinds: "cultural, local",
      type: "cultural",
      distance: `${(i + 1) * 0.8} km`,
      rating: (4 + (i % 5) * 0.18).toFixed(1),
      point: { lat: latitude + (i % 3 - 1) * 0.01, lon: longitude + (i % 5 - 2) * 0.01 },
      lat:   latitude + (i % 3 - 1) * 0.01,
      lon:   longitude + (i % 5 - 2) * 0.01,
      mapsUrl: `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(name + " " + place)}`,
    }));
  return jsonResponse({ ok: true, activities: fallback, source: "fallback", center: { lat: latitude, lon: longitude } });
};
