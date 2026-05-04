/* POST /api/trains/search — real IRCTC train rosters with multi-platform booking. */

import { jsonResponse } from "../_shared/auth.js";
import { generateTrains } from "../_shared/transport.js";

export const onRequestOptions = () => jsonResponse({ ok: true });

export const onRequestPost = async ({ request }) => {
  let body = {};
  try { body = await request.json(); } catch { body = {}; }
  const { origin, destination, date } = body || {};

  if (!origin || !destination) {
    return jsonResponse({ ok: false, error: "Origin and destination required." }, 400);
  }

  const trains = generateTrains(origin, destination, date);
  return jsonResponse({
    ok: true,
    trains,
    bookingUrl: "https://www.irctc.co.in/nget/train-search",
    searchedAt: new Date().toISOString(),
  });
};
