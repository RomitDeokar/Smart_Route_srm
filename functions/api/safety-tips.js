/* POST /api/safety-tips — general + city + persona safety advice. */

import { jsonResponse } from "./_shared/auth.js";
import { getSafetyTips, getEmergencyContacts } from "./_shared/extras.js";

export const onRequestOptions = () => jsonResponse({ ok: true });

export const onRequestPost = async ({ request }) => {
  let body = {};
  try { body = await request.json(); } catch { body = {}; }
  const { destination, persona } = body || {};
  return jsonResponse({
    ok: true,
    tips: getSafetyTips(destination || "", persona || "explorer"),
    emergency: getEmergencyContacts(destination || ""),
  });
};
