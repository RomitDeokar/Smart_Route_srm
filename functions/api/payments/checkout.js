/* POST /api/payments/checkout — mock Stripe checkout (no SDK).
   Returns a fake session URL so the UI can show "Redirecting...". */

import { jsonResponse } from "../_shared/auth.js";

export const onRequestOptions = () => jsonResponse({ ok: true });

export const onRequestPost = async ({ request }) => {
  let body = {};
  try { body = await request.json(); } catch { body = {}; }
  const { amount, description } = body || {};

  const amountInt = Math.round(Number(amount) || 0);
  if (!amountInt) return jsonResponse({ ok: false, error: "Amount is required." }, 400);

  const id = "cs_test_" + crypto.randomUUID().replace(/-/g, "").slice(0, 24);
  return jsonResponse({
    ok: true,
    sessionId: id,
    url: `https://checkout.stripe.com/c/pay/${id}#fidkdWxOYHwnPyd1blppbHNgWg`,
    amount: amountInt,
    currency: "inr",
    description: description || "SmartRoute booking",
  });
};
