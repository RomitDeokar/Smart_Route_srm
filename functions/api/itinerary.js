/* POST /api/itinerary — structured day-by-day plan with real attractions,
   restaurant suggestions, language tips, packing list, real flights, trains,
   hotels, cabs (with real booking URLs) and map coordinates for every stop. */

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
  try { body = await request.json(); } catch { body = {}; }
  const { destination, origin, number_of_days, budget, interests, persona } = body || {};

  if (!destination || !String(destination).trim()) {
    return jsonResponse({ ok: false, error: "Destination is required." }, 400);
  }
  const totalDays = Math.max(1, Math.min(Number(number_of_days) || 3, 10));
  const place     = String(destination).trim();
  const orig      = origin ? String(origin).trim() : "";
  const sel       = Array.isArray(interests) && interests.length ? interests
                                                                 : ["Attractions","Food","Local culture"];

  // Resolve coords + curated attractions (proximity-sorted to actual destination point)
  const cityKey = place.toLowerCase().split(",")[0].trim();
  const geoDest = geocode(place);
  const geoOrig = orig ? geocode(orig) : null;
  const srmKey  = isSRMCity(place);

  // Build attraction list, anchored to actual destination coords so SRM/Kattankulathur
  // pulls nearby ECR / Mahabalipuram POIs first, not Marina Beach 40 km away.
  let curated = getTopAttractions(srmKey === 'chennai' && /srm|kattankulathur/i.test(place) ? 'chennai srm' : (geoDest.resolvedCity || cityKey), geoDest.lat, geoDest.lon);
  if (!curated.length) curated = getTopAttractions(cityKey, geoDest.lat, geoDest.lon);
  if (!curated.length) curated = CITY_TOP_ATTRACTIONS[cityKey] || [];

  // Drop attractions that are absurdly far from the destination anchor (>120 km) —
  // protects users from "Hyderabad → Chennai SRM showing wrong markers" issues
  // when a fuzzy city match returned a list from another region.
  curated = curated.filter(a =>
    typeof a.lat === 'number' && typeof a.lon === 'number' &&
    haversineKm(a.lat, a.lon, geoDest.lat, geoDest.lon) <= 120
  );

  // Enrich attractions with Wikipedia photos + extracts (parallel, time-bounded)
  // Limit to first 24 to stay within Cloudflare Worker subrequest budget.
  try {
    const head = curated.slice(0, 24);
    const tail = curated.slice(24);
    const enriched = await enrichWithPhotos(head, {
      titleKey: "wikiTitle",
      fallbackKey: "name",
      cityForFallback: cityKey,
      concurrency: 8,
    });
    curated = [...enriched, ...tail];
  } catch { /* enrichment is best-effort */ }

  // Live weather (best-effort)
  let weather = [];
  try {
    const raw = await fetchWeather(geoDest.lat, geoDest.lon, totalDays);
    weather = raw.map(d => ({ ...d, emoji: weatherEmoji(d.weatherCode) }));
  } catch { weather = []; }

  // Restaurants — also filter by proximity so they sit near today's stops.
  let restaurants = generateRestaurants(place, geoDest.lat, geoDest.lon);
  if (Array.isArray(restaurants) && restaurants.length && typeof geoDest.lat === 'number') {
    restaurants = [...restaurants].sort((a, b) => {
      const da = haversineKm(a.lat || geoDest.lat, a.lon || geoDest.lon, geoDest.lat, geoDest.lon);
      const db = haversineKm(b.lat || geoDest.lat, b.lon || geoDest.lon, geoDest.lat, geoDest.lon);
      return da - db;
    });
    // Decorate restaurants with Unsplash placeholders (free, deterministic)
    restaurants = restaurants.map(r => ({
      ...r,
      thumbnail: r.thumbnail || unsplashFor(`${r.cuisine || "indian"} food restaurant ${cityKey}`, "600x400"),
      image:     r.image     || unsplashFor(`${r.cuisine || "indian"} food restaurant ${cityKey}`, "1200x800"),
    }));
  }

  const languageTips = getLanguageTips(place);
  const packingList  = generatePackingList(totalDays, weather, persona || "explorer");
  const emergency    = getEmergencyContacts(place);
  const safetyTips   = getSafetyTips(place, persona || "explorer");

  // Real flights / trains / hotels / cabs with real booking URLs
  const today      = new Date().toISOString().split('T')[0];
  let flights = [], trains = [], hotels = [], cabs = [];
  try { flights = orig ? generateFlights(orig, place, today).slice(0, 6) : []; } catch { flights = []; }
  try { trains  = orig ? generateTrains(orig, place).slice(0, 6) : []; } catch { trains = []; }
  try { hotels  = generateHotels(place, totalDays, persona || "explorer").slice(0, 8); } catch { hotels = []; }
  try { cabs    = generateCabs(place).slice(0, 8); } catch { cabs = []; }

  // Hotels & destination hero photo (low-cost decoration)
  hotels = hotels.map(h => ({
    ...h,
    thumbnail: h.thumbnail || h.image || unsplashFor(`${h.name || "hotel"} ${cityKey}`, "600x400"),
    image:     h.image     || unsplashFor(`${h.name || "hotel"} ${cityKey}`, "1200x800"),
  }));
  const heroImage = curated.find(c => c.image)?.image || unsplashFor(`${place} travel landmark`, "1600x900");

  // Distance/route info for the trip header
  const routeKm   = orig ? getDistance(orig, place) : null;
  const oIATA     = orig ? getIATA(orig) : '';
  const dIATA     = getIATA(place);

  const dailyBudget = Math.round((Number(budget) || 18000) / totalDays);
  const days = Array.from({ length: totalDays }, (_, i) => {
    const day = i + 1;
    const len = Math.max(curated.length, 1);
    const idx1 = (i * 3) % len;
    const idx2 = (i * 3 + 1) % len;
    const idx3 = (i * 3 + 2) % len;
    const a1 = curated[idx1];
    const a2 = curated[idx2];
    const a3 = curated[idx3];

    const r1 = restaurants[i % Math.max(restaurants.length, 1)];
    const r2 = restaurants[(i + 2) % Math.max(restaurants.length, 1)];

    const plan = [
      a1 && {
        type: "Attraction", time: "09:00",
        name: a1.name,
        note: a1.extract || a1.description || `Iconic ${a1.type || "spot"} in ${place}.`,
        lat: a1.lat, lon: a1.lon, kind: a1.type || "attraction",
        wikiTitle: a1.wikiTitle, wikiUrl: a1.wikiUrl,
        thumbnail: a1.thumbnail, image: a1.image, extract: a1.extract,
        mapsUrl: `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(a1.name + " " + place)}`,
      },
      r1 && {
        type: "Restaurant", time: "12:30",
        name: r1.name,
        note: `${r1.cuisine} · ${r1.price_range} · ₹${r1.avgCost} per person · ⭐ ${r1.rating}`,
        lat: r1.lat, lon: r1.lon, kind: "restaurant",
        thumbnail: r1.thumbnail, image: r1.image,
        bookingUrl: r1.bookingUrl || r1.zomato || `https://www.zomato.com/${encodeURIComponent(cityKey)}/restaurants`,
        mapsUrl: r1.mapsUrl || `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(r1.name + " " + place)}`,
        cost: r1.avgCost,
      },
      a2 && {
        type: "Attraction", time: "14:30",
        name: a2.name,
        note: a2.extract || a2.description || `Heritage / ${a2.type || "site"} stop.`,
        lat: a2.lat, lon: a2.lon, kind: a2.type || "attraction",
        wikiTitle: a2.wikiTitle, wikiUrl: a2.wikiUrl,
        thumbnail: a2.thumbnail, image: a2.image, extract: a2.extract,
        mapsUrl: `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(a2.name + " " + place)}`,
      },
      a3 && {
        type: "Activity", time: "17:00",
        name: a3.name,
        note: a3.extract || a3.description || `Late-afternoon ${a3.type || "activity"}.`,
        lat: a3.lat, lon: a3.lon, kind: a3.type || "activity",
        wikiTitle: a3.wikiTitle, wikiUrl: a3.wikiUrl,
        thumbnail: a3.thumbnail, image: a3.image, extract: a3.extract,
        mapsUrl: `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(a3.name + " " + place)}`,
      },
      r2 && {
        type: "Dinner", time: "20:00",
        name: r2.name,
        note: `Evening dining · ${r2.cuisine} · ₹${r2.avgCost} per person`,
        lat: r2.lat, lon: r2.lon, kind: "restaurant",
        thumbnail: r2.thumbnail, image: r2.image,
        bookingUrl: r2.bookingUrl || r2.zomato || `https://www.zomato.com/${encodeURIComponent(cityKey)}/restaurants`,
        mapsUrl: r2.mapsUrl || `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(r2.name + " " + place)}`,
        cost: r2.avgCost,
      },
    ].filter(Boolean);

    const w = weather[i] || null;
    return {
      day,
      theme: day === 1 ? "Arrival & orientation"
           : day === totalDays ? "Final exploration & departure"
           : `Day ${day} · ${a1?.type || "Exploration"} loop`,
      dailyBudget,
      weather: w ? {
        emoji: w.emoji, max: w.max, min: w.min, label: w.label,
        precipitation: w.precipitation, weatherCode: w.weatherCode,
      } : null,
      plan,
    };
  });

  // Aggregate ALL stops (with coordinates) for the map view
  const allStops = days.flatMap(d => d.plan.filter(p => p.lat && p.lon).map(p => ({
    day: d.day, ...p,
  })));

  return jsonResponse({
    ok: true,
    itinerary: {
      title: `${place} ${totalDays}-day itinerary`,
      summary: `Built around ${sel.join(", ")} with ₹${Number(budget || 18000).toLocaleString("en-IN")} budget. Weather-aware, real-attractions plan.`,
      destination: place,
      origin: orig || null,
      destCoords: { lat: geoDest.lat, lon: geoDest.lon, name: geoDest.name },
      originCoords: geoOrig ? { lat: geoOrig.lat, lon: geoOrig.lon, name: geoOrig.name } : null,
      isSrmDestination: !!srmKey,
      route: orig ? {
        from: orig, to: place,
        fromIATA: oIATA, toIATA: dIATA,
        distanceKm: routeKm,
      } : null,
      heroImage,
      days,
      allStops,
      weather,
      restaurants: restaurants.slice(0, 8),
      flights,
      trains,
      hotels,
      cabs,
      languageTips,
      packingList,
      emergency,
      safetyTips: safetyTips.slice(0, 12),
      totalAttractions: allStops.filter(s => s.type === "Attraction" || s.type === "Activity").length,
    }
  });
};
