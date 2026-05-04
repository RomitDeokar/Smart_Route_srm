/* POST /api/cabs/search — Ola/Uber/Rapido/BluSmart options. */

import { jsonResponse } from "../_shared/auth.js";
import { getDistance } from "../_shared/cities.js";

export const onRequestOptions = () => jsonResponse({ ok: true });

const PROVIDERS = [
  { name: "Ola",     types: ["Mini","Prime Sedan","Prime SUV"], baseFare: 50, perKm: 14, color: "#16a34a" },
  { name: "Uber",    types: ["UberGo","Premier","UberXL"],      baseFare: 55, perKm: 15, color: "#000000" },
  { name: "Rapido",  types: ["Bike","Auto"],                    baseFare: 25, perKm: 7,  color: "#FFCA08" },
  { name: "BluSmart",types: ["EV Sedan","EV Premium"],          baseFare: 60, perKm: 13, color: "#0EA5E9" },
  { name: "Meru",    types: ["Sedan","SUV"],                    baseFare: 70, perKm: 16, color: "#DC2626" },
  { name: "InDrive", types: ["Bid"],                            baseFare: 40, perKm: 12, color: "#6366F1" },
];
const CITY_MULTIPLIER = { mumbai:1.4, delhi:1.2, bangalore:1.15, chennai:1.1, hyderabad:1.05 };

export const onRequestPost = async ({ request }) => {
  let body = {};
  try { body = await request.json(); } catch { body = {}; }
  const { origin, destination, city } = body || {};
  if (!origin || !destination) {
    return jsonResponse({ ok: false, error: "Origin and destination required." }, 400);
  }
  const dist = Math.min(getDistance(origin, destination), 80);
  const cityKey = String(city || destination || "").toLowerCase().replace(/[^a-z]/g,"");
  let mult = 1;
  for (const [k,m] of Object.entries(CITY_MULTIPLIER)) {
    if (cityKey.includes(k)) { mult = m; break; }
  }

  const options = PROVIDERS.flatMap((p, pi) =>
    p.types.map((t, ti) => {
      const fare = Math.round((p.baseFare + dist * p.perKm) * mult * (1 + ti * 0.25));
      const eta  = `${4 + ((pi + ti) % 8)} min`;
      const oEnc = encodeURIComponent(origin), dEnc = encodeURIComponent(destination);
      return {
        id: `CAB${pi}${ti}`,
        provider: p.name, type: t,
        price: fare, currency: "₹", eta,
        rating: (4.0 + ((pi + ti) % 9) / 10).toFixed(1),
        bookingUrl: p.name === "Ola"     ? `https://www.olacabs.com/?pickup=${oEnc}&drop=${dEnc}`
                   : p.name === "Uber"   ? `https://m.uber.com/ul/?action=setPickup&pickup[formatted_address]=${oEnc}&dropoff[formatted_address]=${dEnc}`
                   : p.name === "Rapido" ? `https://www.rapido.bike`
                   : p.name === "BluSmart"? `https://www.blu-smart.com/`
                   : p.name === "Meru"   ? `https://www.meru.in/`
                   : `https://indrive.com/`,
        mapsUrl: `https://www.google.com/maps/dir/?api=1&origin=${oEnc}&destination=${dEnc}`,
      };
    })
  ).sort((a,b) => a.price - b.price);

  return jsonResponse({ ok: true, cabs: options, distance_km: dist });
};
