/* POST /api/flights/search — full GitHub-port real-airline + multi-platform booking. */

import { jsonResponse } from "../_shared/auth.js";
import { generateFlights } from "../_shared/transport.js";

export const onRequestOptions = () => jsonResponse({ ok: true });

export const onRequestPost = async ({ request }) => {
  let body = {};
  try { body = await request.json(); } catch { body = {}; }
  const { origin, destination, departure_date, passengers } = body || {};

  if (!origin || !destination) {
    return jsonResponse({ ok: false, error: "Origin and destination required." }, 400);
  }

  const pax = Math.max(1, Number(passengers) || 1);
  const dateParam = departure_date || new Date(Date.now() + 7 * 86400000).toISOString().split("T")[0];

  const raw = generateFlights(origin, destination, dateParam);
  // Reshape for the existing UI (provide both legacy + new fields).
  const flights = raw.map(f => {
    const total = f.price * pax;
    return {
      ...f,
      airlineCode: f.flight_no?.split(" ")[0] || "",
      flightNo:    f.flight_no,
      origin, destination,
      departureTime: f.departure,
      arrivalTime:   f.arrival,
      price:         total,
      perPerson:     `₹${f.price.toLocaleString("en-IN")}`,
      priceFormatted:`₹${total.toLocaleString("en-IN")}`,
      stops:         f.stops === 0 ? "Non-stop" : `${f.stops} stop`,
      class:         f.class,
      bookingUrl:    f.bookingPlatforms?.[0]?.url,
      bookingPlatforms: f.bookingPlatforms,
      refundable:    f.class !== "Economy",
    };
  });

  return jsonResponse({
    ok: true, flights,
    bookingUrl: `https://www.google.com/travel/flights?q=flights+from+${encodeURIComponent(origin)}+to+${encodeURIComponent(destination)}`,
    searchedAt: new Date().toISOString(),
  });
};
