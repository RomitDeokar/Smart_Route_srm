/* POST /api/multi-city — Multi-city itinerary generator that chains the
   single-city planner across N destinations with proper inter-city legs,
   per-city day allocation and an aggregated map. Ports the
   /api/generate-multi-city feature from the original GitHub backend. */

import { jsonResponse } from "./_shared/auth.js";
import { CITY_TOP_ATTRACTIONS, getTopAttractions, geocode, haversineKm, getDistance, getIATA } from "./_shared/cities.js";
import { generateRestaurants, getLanguageTips, generatePackingList, getEmergencyContacts, getSafetyTips } from "./_shared/extras.js";
import { fetchWeather, weatherEmoji } from "./_shared/data.js";
import { isSRMCity } from "./_shared/srm.js";
import { generateFlights, generateTrains } from "./_shared/transport.js";
import { generateHotels, generateCabs } from "./_shared/hotels-real.js";
import { enrichWithPhotos, unsplashFor } from "./_shared/wiki.js";

export const onRequestOptions = () => jsonResponse({ ok: true });

export const onRequestPost = async ({ request }) => {
  let body = {};
  try { body = await request.json(); } catch {}
  const { cities = [], daysPerCity = 2, budget = 30000, persona = "explorer", origin = "" } = body || {};

  if (!Array.isArray(cities) || cities.length === 0) {
    return jsonResponse({ ok: false, error: "At least one city required in `cities` array." }, 400);
  }
  if (cities.length > 6) {
    return jsonResponse({ ok: false, error: "Maximum 6 cities per trip." }, 400);
  }

  const totalDays = cities.length * Math.max(1, Math.min(daysPerCity, 5));
  const perCityBudget = Math.round(Number(budget) / cities.length);
  const today = new Date().toISOString().split("T")[0];

  const segments = [];
  let prev = origin || cities[0];

  for (let ci = 0; ci < cities.length; ci++) {
    const place = String(cities[ci]).trim();
    const cityKey = place.toLowerCase().split(",")[0].trim();
    const geo = geocode(place);
    const srmKey = isSRMCity(place);

    let curated = getTopAttractions(srmKey === "chennai" && /srm|kattankulathur/i.test(place) ? "chennai srm" : (geo.resolvedCity || cityKey), geo.lat, geo.lon);
    if (!curated.length) curated = CITY_TOP_ATTRACTIONS[cityKey] || [];
    curated = curated.filter(a =>
      typeof a.lat === "number" && typeof a.lon === "number" &&
      haversineKm(a.lat, a.lon, geo.lat, geo.lon) <= 120
    );

    try {
      const head = curated.slice(0, 16);
      const tail = curated.slice(16);
      const enriched = await enrichWithPhotos(head, {
        titleKey: "wikiTitle", fallbackKey: "name", cityForFallback: cityKey, concurrency: 6,
      });
      curated = [...enriched, ...tail];
    } catch {}

    let weather = [];
    try {
      const raw = await fetchWeather(geo.lat, geo.lon, daysPerCity);
      weather = raw.map(d => ({ ...d, emoji: weatherEmoji(d.weatherCode) }));
    } catch {}

    let restaurants = generateRestaurants(place, geo.lat, geo.lon).slice(0, 6).map(r => ({
      ...r,
      thumbnail: r.thumbnail || unsplashFor(`${r.cuisine || "indian"} food ${cityKey}`, "600x400"),
    }));

    let hotels = [];
    try { hotels = generateHotels(place, daysPerCity, persona).slice(0, 4); } catch {}
    hotels = hotels.map(h => ({
      ...h,
      thumbnail: h.thumbnail || unsplashFor(`${h.name || "hotel"} ${cityKey}`, "600x400"),
    }));

    let intercityFlights = [], intercityTrains = [];
    if (prev && prev.toLowerCase() !== place.toLowerCase()) {
      try { intercityFlights = generateFlights(prev, place, today).slice(0, 3); } catch {}
      try { intercityTrains = generateTrains(prev, place).slice(0, 3); } catch {}
    }

    const dailyBudget = Math.round(perCityBudget / Math.max(1, daysPerCity));
    const days = Array.from({ length: daysPerCity }, (_, i) => {
      const day = i + 1;
      const len = Math.max(curated.length, 1);
      const a1 = curated[(i * 3) % len];
      const a2 = curated[(i * 3 + 1) % len];
      const a3 = curated[(i * 3 + 2) % len];
      const r1 = restaurants[i % Math.max(restaurants.length, 1)];
      const plan = [
        a1 && {
          type: "Attraction", time: "09:00", name: a1.name,
          note: a1.extract || a1.description || `Top ${a1.type || "attraction"}`,
          lat: a1.lat, lon: a1.lon, kind: a1.type || "attraction",
          thumbnail: a1.thumbnail, image: a1.image, extract: a1.extract,
          wikiTitle: a1.wikiTitle, wikiUrl: a1.wikiUrl,
          mapsUrl: `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(a1.name + " " + place)}`,
        },
        r1 && {
          type: "Restaurant", time: "13:00", name: r1.name,
          note: `${r1.cuisine} · ₹${r1.avgCost} pp · ⭐ ${r1.rating}`,
          lat: r1.lat, lon: r1.lon, kind: "restaurant",
          thumbnail: r1.thumbnail, image: r1.image,
          mapsUrl: r1.mapsUrl,
        },
        a2 && {
          type: "Attraction", time: "15:00", name: a2.name,
          note: a2.extract || a2.description || `Heritage stop`,
          lat: a2.lat, lon: a2.lon, kind: a2.type || "attraction",
          thumbnail: a2.thumbnail, image: a2.image, extract: a2.extract,
          wikiTitle: a2.wikiTitle, wikiUrl: a2.wikiUrl,
        },
        a3 && {
          type: "Activity", time: "18:00", name: a3.name,
          note: a3.extract || a3.description || `Evening activity`,
          lat: a3.lat, lon: a3.lon,
          thumbnail: a3.thumbnail, image: a3.image, extract: a3.extract,
          wikiTitle: a3.wikiTitle, wikiUrl: a3.wikiUrl,
        },
      ].filter(Boolean);
      const w = weather[i] || null;
      return {
        day, cityIndex: ci, city: place,
        theme: day === 1 ? `Arrive in ${place}` : day === daysPerCity ? `Final day in ${place}` : `Day ${day} · ${place}`,
        dailyBudget,
        weather: w ? { emoji: w.emoji, max: w.max, min: w.min, label: w.label, precipitation: w.precipitation } : null,
        plan,
      };
    });

    const heroImage = curated.find(c => c.image)?.image || unsplashFor(`${place} travel landmark`, "1600x900");

    segments.push({
      city: place,
      coords: { lat: geo.lat, lon: geo.lon, name: geo.name },
      isSrm: !!srmKey,
      heroImage,
      previousCity: prev !== place ? prev : null,
      legDistanceKm: prev !== place ? getDistance(prev, place) : 0,
      intercityFlights,
      intercityTrains,
      hotels,
      restaurants,
      weather,
      days,
      cityBudget: perCityBudget,
    });

    prev = place;
  }

  // Aggregate stops with city tag for unified map
  const allStops = segments.flatMap(s => s.days.flatMap(d => d.plan.filter(p => p.lat && p.lon).map(p => ({
    city: s.city, day: d.day, ...p,
  }))));

  return jsonResponse({
    ok: true,
    multiCity: {
      title: `${cities.length}-city tour: ${cities.join(" → ")}`,
      summary: `${totalDays} days across ${cities.length} cities · ₹${Number(budget).toLocaleString("en-IN")} total budget · ${persona} persona.`,
      origin: origin || null,
      cities,
      totalDays,
      perCityBudget,
      segments,
      allStops,
      route: cities.map((c, i) => ({
        from: i === 0 ? origin || c : cities[i - 1],
        to: c,
        iata: getIATA(c),
        distanceKm: i === 0 ? (origin ? getDistance(origin, c) : 0) : getDistance(cities[i - 1], c),
      })),
    },
  });
};
