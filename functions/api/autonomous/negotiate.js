/* POST /api/autonomous/negotiate — Autonomous negotiator agent.
   Simulates bargaining for hotel/cab/activity discounts using a
   Bayesian "willingness-to-discount" prior + iterative back-and-forth. */

import { jsonResponse } from "../_shared/auth.js";

export const onRequestOptions = () => jsonResponse({ ok: true });

// Vendor prior: each vendor has an internal {min, max} acceptable discount band.
const VENDOR_PRIOR = {
  hotel:    { min: 0.04, max: 0.18, base: 0.08 },
  cab:      { min: 0.03, max: 0.15, base: 0.06 },
  activity: { min: 0.05, max: 0.20, base: 0.10 },
  flight:   { min: 0.02, max: 0.08, base: 0.03 },
};

export const onRequestPost = async ({ request }) => {
  let body = {};
  try { body = await request.json(); } catch { body = {}; }
  const { items = [], persona = "explorer", urgency = "normal" } = body;

  if (!Array.isArray(items) || items.length === 0) {
    return jsonResponse({ ok: false, error: "items[] required" }, 400);
  }

  // Persona modifier: luxury & student push harder
  const personaMul = persona === "luxury" ? 1.15
                   : persona === "student" ? 1.20
                   : persona === "family" ? 1.08
                   : 1.00;
  const urgencyMul = urgency === "high" ? 0.85 : urgency === "low" ? 1.10 : 1.00;

  const negotiations = items.map(it => {
    const cat = (it.category || it.type || "hotel").toLowerCase();
    const prior = VENDOR_PRIOR[cat] || VENDOR_PRIOR.hotel;
    const price = Number(it.price || it.amount || 0);

    // Iterative bargaining (5 rounds)
    let offered = prior.base;
    const rounds = [];
    for (let r = 0; r < 5; r++) {
      const target = Math.min(prior.max, offered * (1 + 0.18 * personaMul * urgencyMul));
      const counter = Math.max(prior.min, offered + (target - offered) * 0.55);
      rounds.push({ round: r + 1, offered: Number(offered.toFixed(3)), counter: Number(counter.toFixed(3)) });
      offered = counter;
    }
    const finalDiscount = Math.min(prior.max, offered);
    const saved = Math.round(price * finalDiscount);
    return {
      itemId: it.id || `${cat}-${Math.random().toString(36).slice(2, 7)}`,
      name: it.name || `${cat} item`,
      category: cat,
      originalPrice: price,
      finalDiscount: Number(finalDiscount.toFixed(3)),
      discountPct: `${Math.round(finalDiscount * 100)}%`,
      saved,
      finalPrice: Math.max(0, price - saved),
      rounds,
    };
  });

  const totalSaved = negotiations.reduce((s, n) => s + n.saved, 0);
  const totalOriginal = negotiations.reduce((s, n) => s + n.originalPrice, 0);

  return jsonResponse({
    ok: true,
    mode: "autonomous-negotiate",
    persona,
    urgency,
    method: "Bayesian willingness-to-discount + 5-round iterative bargaining",
    negotiations,
    totals: {
      original: totalOriginal,
      saved: totalSaved,
      finalSpend: totalOriginal - totalSaved,
      avgDiscount: totalOriginal > 0 ? Number((totalSaved / totalOriginal).toFixed(3)) : 0,
    },
    recommendation: totalSaved > 0
      ? `Negotiator unlocked ₹${totalSaved.toLocaleString("en-IN")} (${Math.round(totalSaved/Math.max(totalOriginal,1)*100)}%) across ${negotiations.length} items.`
      : "No further discount available — pricing already optimal.",
  });
};
