/* POST /api/recommendations — destinations matching budget + preferences. */

import { jsonResponse } from "./_shared/auth.js";
import { getRecommendations } from "./_shared/extras.js";

export const onRequestOptions = () => jsonResponse({ ok: true });

export const onRequestPost = async ({ request }) => {
  let body = {};
  try { body = await request.json(); } catch { body = {}; }
  const { budget, duration, preferences, currentLocation } = body || {};
  const recs = getRecommendations(
    Number(budget) || 20000,
    Number(duration) || 4,
    Array.isArray(preferences) ? preferences : [],
    currentLocation || null
  );
  return jsonResponse({ ok: true, recommendations: recs });
};
