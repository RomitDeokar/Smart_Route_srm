/* POST /api/compare-trips — Side-by-side comparison of multiple itinerary
   options. Ports the /api/compare-trips feature from the original GitHub
   backend. Accepts an array of mini-trip configs and returns scored
   summaries so the UI can render a comparison grid. */

import { jsonResponse } from "./_shared/auth.js";
import { geocode, haversineKm, getDistance, getTopAttractions } from "./_shared/cities.js";
import { fetchWeather, weatherEmoji } from "./_shared/data.js";
import { generateHotels } from "./_shared/hotels-real.js";
import { generateFlights, generateTrains } from "./_shared/transport.js";

export const onRequestOptions = () => jsonResponse({ ok: true });

function scoreTrip(t) {
  // Simple multi-criteria score: budget fit, weather, attractions, transport.
  const budgetFit = Math.min(1, 30000 / Math.max(t.budget || 1, 1));
  const wxOk = t.weatherOk ? 1 : 0.55;
  const attractions = Math.min(1, (t.attractionsCount || 0) / 8);
  const transport = (t.flightsCount > 0 ? 0.5 : 0) + (t.trainsCount > 0 ? 0.5 : 0);
  const composite = (budgetFit * 0.30) + (wxOk * 0.25) + (attractions * 0.25) + (transport * 0.20);
  return Math.round(composite * 100);
}

export const onRequestPost = async ({ request }) => {
  let body = {};
  try { body = await request.json(); } catch {}
  const { trips = [] } = body || {};
  if (!Array.isArray(trips) || trips.length < 2) {
    return jsonResponse({ ok: false, error: "Provide at least 2 trips to compare in `trips` array." }, 400);
  }
  if (trips.length > 5) {
    return jsonResponse({ ok: false, error: "Compare up to 5 trips at once." }, 400);
  }

  const today = new Date().toISOString().split("T")[0];
  const summaries = [];

  for (const t of trips) {
    const dest = String(t.destination || "").trim();
    const orig = String(t.origin || "").trim();
    if (!dest) { summaries.push({ ok: false, error: "missing destination", input: t }); continue; }
    const days = Math.max(1, Math.min(Number(t.days || 3), 10));
    const budget = Number(t.budget || 20000);
    const persona = t.persona || "explorer";

    const geo = geocode(dest);
    let attractions = getTopAttractions(geo.resolvedCity || dest.toLowerCase(), geo.lat, geo.lon)
      .filter(a => typeof a.lat === "number" && haversineKm(a.lat, a.lon, geo.lat, geo.lon) <= 120)
      .slice(0, 12);

    let weather = [];
    try {
      const raw = await fetchWeather(geo.lat, geo.lon, days);
      weather = raw.map(d => ({ ...d, emoji: weatherEmoji(d.weatherCode) }));
    } catch {}
    const weatherOk = weather.every(w => (w.precipitation || 0) < 60);

    let flights = [], trains = [], hotels = [];
    try { flights = orig ? generateFlights(orig, dest, today).slice(0, 3) : []; } catch {}
    try { trains  = orig ? generateTrains(orig, dest).slice(0, 3) : []; } catch {}
    try { hotels  = generateHotels(dest, days, persona).slice(0, 3); } catch {}

    const cheapestFlight = flights.length ? Math.min(...flights.map(f => Number(f.price) || 99999)) : null;
    const cheapestTrain  = trains.length  ? Math.min(...trains.map(t => Number(t.price) || 99999))  : null;
    const cheapestHotel  = hotels.length  ? Math.min(...hotels.map(h => Number(h.pricePerNight || h.price) || 9999)) : null;

    const meta = {
      destination: dest,
      origin: orig || null,
      days,
      budget,
      persona,
      attractionsCount: attractions.length,
      flightsCount: flights.length,
      trainsCount: trains.length,
      hotelsCount: hotels.length,
      cheapestFlight,
      cheapestTrain,
      cheapestHotel,
      avgTempMax: weather.length ? Math.round(weather.reduce((s,w)=>s+(w.max||0),0)/weather.length) : null,
      avgTempMin: weather.length ? Math.round(weather.reduce((s,w)=>s+(w.min||0),0)/weather.length) : null,
      maxRainPct: weather.length ? Math.max(...weather.map(w=>w.precipitation||0)) : 0,
      weatherOk,
      distanceKm: orig ? getDistance(orig, dest) : null,
      heroImage: attractions.find(a => a.image)?.image || null,
    };
    meta.score = scoreTrip(meta);
    meta.recommendation =
      meta.score >= 80 ? "✅ Excellent fit — highly recommended" :
      meta.score >= 65 ? "👍 Good option — solid balance" :
      meta.score >= 50 ? "🤔 Decent — some trade-offs" :
                         "⚠ Reconsider — may not fit budget/weather window";
    summaries.push(meta);
  }

  // Rank by score
  const ranked = [...summaries]
    .filter(s => !s.error)
    .sort((a, b) => (b.score || 0) - (a.score || 0));

  return jsonResponse({
    ok: true,
    comparison: {
      count: summaries.length,
      summaries,
      best: ranked[0] || null,
      worst: ranked[ranked.length - 1] || null,
      generatedAt: new Date().toISOString(),
    },
  });
};
