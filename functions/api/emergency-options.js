/* POST /api/emergency-options — emergency replanning. */

import { jsonResponse } from "./_shared/auth.js";

export const onRequestOptions = () => jsonResponse({ ok: true });

export const onRequestPost = async ({ request }) => {
  let body = {};
  try { body = await request.json(); } catch { body = {}; }
  const { origin, destination } = body || {};
  const dest = destination || "Destination";
  const orig = origin || "Origin";

  return jsonResponse({
    ok: true,
    options: {
      nearbyHotels: [
        `${dest} Airport Transit Hotel`,
        `${dest} Budget Inn (24hr check-in)`,
        `${dest} City Centre Emergency Stay`,
      ],
      alternateFlights: [
        `${orig} → ${dest} redeye (next day)`,
        `${orig} → ${dest} via Delhi connection`,
        `${orig} → ${dest} morning first flight`,
      ],
      transportOptions: [
        "Ola/Uber priority booking",
        "Railway station transfer",
        "Pre-paid airport taxi",
        "Local auto-rickshaw",
      ],
      emergencyContacts: {
        police: "100", ambulance: "108",
        touristHelpline: "1800-111-363",
        airportHelpdesk: "1800-180-1407",
      },
      insuranceTips: [
        "File a delay certificate at the airline counter",
        "Photograph all receipts for reimbursement",
        "Contact your bank for emergency card limit increase",
      ]
    }
  });
};
