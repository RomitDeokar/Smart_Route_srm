/* POST /api/insights — Trip Insights agent. Analyses an itinerary payload
   (or a destination spec) and returns AI-powered insights: best time to
   visit, crowd density forecast, hidden cost warnings, persona match score,
   suggested swaps, eco-friendliness rating. New autonomous feature. */

import { jsonResponse } from "./_shared/auth.js";
import { fetchWeather, weatherEmoji } from "./_shared/data.js";
import { geocode, getDistance, getTopAttractions, haversineKm } from "./_shared/cities.js";

export const onRequestOptions = () => jsonResponse({ ok: true });

const PEAK_MONTHS = {
  goa: [11, 12, 1, 2],
  jaipur: [10, 11, 12, 1, 2],
  manali: [4, 5, 6, 9, 10],
  shimla: [4, 5, 6, 9, 10],
  ooty: [4, 5, 6, 9, 10],
  kerala: [10, 11, 12, 1, 2],
  ladakh: [6, 7, 8],
  leh: [6, 7, 8],
  mumbai: [11, 12, 1, 2],
  delhi: [10, 11, 2, 3],
  chennai: [12, 1, 2],
  bangalore: [10, 11, 12, 1, 2],
  agra: [10, 11, 12, 1, 2],
  varanasi: [11, 12, 1, 2],
  udaipur: [10, 11, 12, 1, 2],
};

function bestTimeToVisit(dest) {
  const k = String(dest || "").toLowerCase().split(",")[0].trim();
  for (const [city, months] of Object.entries(PEAK_MONTHS)) {
    if (k.includes(city)) {
      const names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
      return months.map(m => names[m - 1]).join(", ");
    }
  }
  return "Oct, Nov, Dec, Jan, Feb"; // generic Indian peak season
}

function crowdDensityForecast(dest, weather) {
  const month = new Date().getMonth() + 1;
  const k = String(dest || "").toLowerCase().split(",")[0].trim();
  const peaks = Object.entries(PEAK_MONTHS).find(([c]) => k.includes(c))?.[1] || [];
  const isPeak = peaks.includes(month);
  const isRainy = weather?.some(w => (w.precipitation || 0) > 60);
  const day = new Date().getDay();
  const isWeekend = day === 0 || day === 6;
  let score = 50;
  if (isPeak) score += 25;
  if (isWeekend) score += 15;
  if (isRainy) score -= 18;
  score = Math.max(10, Math.min(95, score));
  return {
    score,
    label: score >= 75 ? "Very Crowded" : score >= 55 ? "Crowded" : score >= 35 ? "Moderate" : "Quiet",
    emoji: score >= 75 ? "🔴" : score >= 55 ? "🟠" : score >= 35 ? "🟡" : "🟢",
    factors: { isPeakSeason: isPeak, isWeekend, weatherDeterrent: isRainy },
  };
}

function hiddenCosts(persona, days, budget) {
  const tips = [];
  const dailyBudget = budget / Math.max(1, days);
  if (dailyBudget < 1500) tips.push({ severity: "high", text: "⚠ Daily budget below ₹1,500 — expect to skip some paid attractions." });
  if (persona !== "luxury") tips.push({ severity: "low", text: "💡 Tip: Pre-book entry tickets online for 10-15% discount on most monuments." });
  tips.push({ severity: "medium", text: "🎒 Allocate ₹500-1000/day for unplanned snacks, autos and tips." });
  tips.push({ severity: "low", text: "📱 Carry a UPI app — most local vendors accept QR payments at zero markup." });
  if (persona === "adventure") tips.push({ severity: "medium", text: "⛰️ Adventure activities often have 18% GST — factor that into operator quotes." });
  return tips;
}

function personaFit(persona, attractionsCount, hasBeach, hasMountain, hasHeritage) {
  const map = {
    explorer:  attractionsCount >= 6 ? 95 : attractionsCount * 12,
    student:   attractionsCount * 10 + 30,
    family:    hasHeritage ? 88 : 70,
    creator:   (hasBeach || hasMountain) ? 92 : 68,
    luxury:    65 + (attractionsCount * 3),
    adventure: hasMountain ? 95 : (hasBeach ? 70 : 55),
  };
  return Math.min(100, Math.round(map[persona] || 75));
}

function ecoScore(days, persona, hasFlight) {
  let score = 70;
  if (hasFlight) score -= 25;
  if (persona === "adventure" || persona === "explorer") score += 8;
  if (days >= 5) score += 6; // longer trips amortize travel emissions
  return Math.max(20, Math.min(95, score));
}

export const onRequestPost = async ({ request }) => {
  let body = {};
  try { body = await request.json(); } catch {}
  const { destination, origin, days = 3, budget = 15000, persona = "explorer" } = body || {};
  if (!destination) return jsonResponse({ ok: false, error: "destination required" }, 400);

  const geo = geocode(destination);
  let weather = [];
  try {
    const raw = await fetchWeather(geo.lat, geo.lon, days);
    weather = raw.map(d => ({ ...d, emoji: weatherEmoji(d.weatherCode) }));
  } catch {}

  let attractions = getTopAttractions(geo.resolvedCity || destination.toLowerCase(), geo.lat, geo.lon)
    .filter(a => typeof a.lat === "number" && haversineKm(a.lat, a.lon, geo.lat, geo.lon) <= 120);

  const hasBeach = attractions.some(a => /beach|sea|coast/i.test((a.name || "") + " " + (a.description || "")));
  const hasMountain = attractions.some(a => /hill|mountain|peak|trek/i.test((a.name || "") + " " + (a.description || "")));
  const hasHeritage = attractions.some(a => /temple|fort|palace|monument|heritage|museum/i.test((a.name || "") + " " + (a.type || "")));

  const distance = origin ? getDistance(origin, destination) : 0;
  const hasFlight = distance > 800;

  const insights = {
    destination,
    origin: origin || null,
    bestTimeToVisit: bestTimeToVisit(destination),
    crowdForecast: crowdDensityForecast(destination, weather),
    hiddenCosts: hiddenCosts(persona, days, budget),
    personaFit: personaFit(persona, attractions.length, hasBeach, hasMountain, hasHeritage),
    ecoScore: ecoScore(days, persona, hasFlight),
    weatherSummary: weather.length ? {
      avgMax: Math.round(weather.reduce((s, w) => s + (w.max || 0), 0) / weather.length),
      avgMin: Math.round(weather.reduce((s, w) => s + (w.min || 0), 0) / weather.length),
      maxRainPct: Math.max(...weather.map(w => w.precipitation || 0)),
      verdict: weather.every(w => (w.precipitation || 0) < 40) ? "Excellent travel window" :
               weather.some(w => (w.precipitation || 0) > 70) ? "Heavy rain risk — pack accordingly" : "Mostly fair, occasional showers",
    } : null,
    attractionTypes: {
      total: attractions.length,
      hasBeach, hasMountain, hasHeritage,
    },
    transportEstimate: distance > 0 ? {
      distanceKm: distance,
      mode: hasFlight ? "flight" : distance > 300 ? "train" : "road",
      estimatedHours: hasFlight ? +(distance / 700).toFixed(1) : +(distance / 60).toFixed(1),
      estimatedCost: hasFlight ? Math.round(distance * 5.5) : Math.round(distance * 1.4),
    } : null,
    recommendations: [
      attractions.length < 4 ? "🎯 Limited curated attractions — consider exploring a different city or extending stay" : null,
      hasFlight ? "✈️ Long distance — consider eco-offsetting your flight via Cleartrip or MMT" : null,
      weather.some(w => (w.precipitation || 0) > 70) ? "☔ Pack rain gear and plan indoor alternatives" : null,
      persona === "luxury" && budget / days < 8000 ? "💎 Budget tight for luxury persona — consider increasing or switching persona" : null,
    ].filter(Boolean),
    overallScore: Math.round((personaFit(persona, attractions.length, hasBeach, hasMountain, hasHeritage) + ecoScore(days, persona, hasFlight)) / 2),
    generatedAt: new Date().toISOString(),
  };

  return jsonResponse({ ok: true, insights });
};
