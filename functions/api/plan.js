/* POST /api/plan — multi-agent trip planner. */

import { jsonResponse } from "./_shared/auth.js";
import { geocodePlace, fetchWeather, weatherEmoji } from "./_shared/data.js";
import { buildMockPlan } from "./_shared/planner.js";

export const onRequestOptions = () => jsonResponse({ ok: true });

export const onRequestPost = async ({ request }) => {
  let input = {};
  try { input = await request.json(); } catch { input = {}; }

  const originQ = input.origin || "Maraimalai Nagar, Chennai";
  const destQ   = input.destination || "Shillong";
  const days    = Number(input.days) || 5;

  try {
    const [originGeo, destGeo] = await Promise.all([
      geocodePlace(originQ).catch(() => null),
      geocodePlace(destQ).catch(() => null),
    ]);

    const rawWeather = destGeo
      ? await fetchWeather(destGeo.latitude, destGeo.longitude, days).catch(() => [])
      : [];
    const weather = rawWeather.map(d => ({ ...d, emoji: weatherEmoji(d.weatherCode) }));

    const hasRain = weather.some(d => ["🌧️","⛈️","🌦️"].includes(d.emoji));
    const packing = hasRain
      ? ["Rain jacket","Waterproof shoes","Power bank","Quick-dry layer","Umbrella"]
      : ["Light layers","Walking shoes","Power bank","Reusable bottle","Sunscreen"];

    const plan = buildMockPlan(input, {
      geocode: { origin: originGeo, destination: destGeo },
      weather,
      packing,
    });

    return jsonResponse({ ok: true, mode: "mock-ai", plan });
  } catch (e) {
    return jsonResponse({ ok: false, error: e?.message || "Plan generation failed." }, 500);
  }
};
