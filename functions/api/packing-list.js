/* POST /api/packing-list — AI-generated packing list. */

import { jsonResponse } from "./_shared/auth.js";
import { geocodePlace, fetchWeather, weatherEmoji } from "./_shared/data.js";

export const onRequestOptions = () => jsonResponse({ ok: true });

const ESSENTIALS = ["Phone charger","Power bank","ID/Aadhaar","Wallet","Reusable water bottle","Toiletries","First-aid kit"];
const COLD       = ["Warm jacket","Thermal innerwear","Gloves","Beanie","Wool socks","Fleece"];
const HOT        = ["Sunscreen","Sunglasses","Hat/Cap","Light cotton clothes","Hand fan"];
const RAIN       = ["Rain jacket","Waterproof shoes","Umbrella","Quick-dry layer","Plastic ziplock"];

export const onRequestPost = async ({ request }) => {
  let body = {};
  try { body = await request.json(); } catch { body = {}; }
  const { destination, persona, days } = body || {};

  if (!destination || !String(destination).trim())
    return jsonResponse({ ok: false, error: "Destination is required." }, 400);

  const geo = await geocodePlace(destination).catch(() => null);
  const weather = geo
    ? await fetchWeather(geo.latitude, geo.longitude, Number(days) || 5).catch(() => [])
    : [];

  const items = [...ESSENTIALS];
  const avgMax = weather.length ? weather.reduce((s, d) => s + (d.max || 0), 0) / weather.length : 25;
  const avgMin = weather.length ? weather.reduce((s, d) => s + (d.min || 0), 0) / weather.length : 18;
  const rain   = weather.filter(d => (d.precipitation || 0) > 40).length;

  if (avgMin < 12) items.push(...COLD);
  if (avgMax > 32) items.push(...HOT);
  if (rain >= 1)   items.push(...RAIN);

  if (persona === "creator")   items.push("Camera + lenses","Gimbal","SD cards");
  if (persona === "adventure") items.push("Trekking shoes","Waterproof bag","Headlamp");
  if (persona === "family")    items.push("Snacks","Kids' entertainment","Travel pillow");
  if (persona === "luxury")    items.push("Formal outfit","Dress shoes","Eye mask");

  const seen = new Set();
  const dedup = items.filter(i => { if (seen.has(i)) return false; seen.add(i); return true; });

  return jsonResponse({
    ok: true,
    items: dedup,
    destination,
    weatherSummary: {
      avgMax: Math.round(avgMax), avgMin: Math.round(avgMin),
      rainDays: rain,
      forecast: weather.slice(0, 5).map(d => ({ ...d, emoji: weatherEmoji(d.weatherCode) }))
    }
  });
};
