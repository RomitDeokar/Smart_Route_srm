/* POST /api/risk-score — travel risk score for a destination. */

import { jsonResponse } from "./_shared/auth.js";
import { geocodePlace, fetchWeather, weatherEmoji } from "./_shared/data.js";

export const onRequestOptions = () => jsonResponse({ ok: true });

export const onRequestPost = async ({ request }) => {
  let body = {};
  try { body = await request.json(); } catch { body = {}; }
  const { destination, days } = body || {};
  if (!destination || !String(destination).trim())
    return jsonResponse({ ok: false, error: "Destination is required." }, 400);

  const geo = await geocodePlace(destination).catch(() => null);
  const weather = geo
    ? await fetchWeather(geo.latitude, geo.longitude, Number(days) || 5).catch(() => [])
    : [];
  const wW = weather.map(d => ({ ...d, emoji: weatherEmoji(d.weatherCode) }));

  const totalDays = wW.length || Number(days) || 5;
  const rainDays  = wW.filter(d => (d.precipitation || 0) > 50).length;
  const windyDays = wW.filter(d => (d.windSpeed || 0) > 30).length;
  const hotDays   = wW.filter(d => (d.max || 0) > 38).length;

  const score = Math.max(0, 1 - (rainDays * 0.20 + windyDays * 0.10 + hotDays * 0.10));
  const level = score > 0.85 ? "Low" : score > 0.65 ? "Moderate" : "High";

  return jsonResponse({
    ok: true,
    riskScore: {
      level, score: Number(score.toFixed(3)),
      destination, totalDays,
      breakdown: { rainDays, windyDays, hotDays },
      recommendation: level === "High"
        ? "Consider rescheduling or pack heavy weather gear + indoor backups."
        : level === "Moderate"
          ? "Plan with 1-2 backup activities for risky days."
          : "Conditions look great for an outdoor-heavy itinerary.",
      forecast: wW.slice(0, 7),
    }
  });
};
