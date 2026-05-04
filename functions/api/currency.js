/* GET /api/currency?from=INR&to=USD&amount=10000 — Live currency converter
   using exchangerate.host (free, no key) with hardcoded fallback rates so
   the UI always renders something even if the upstream is down. */

import { jsonResponse } from "./_shared/auth.js";

// Fallback rates against INR (base) — last reviewed May 2025.
const FALLBACK_RATES = {
  INR: 1.0,    USD: 0.012,  EUR: 0.011, GBP: 0.0094, JPY: 1.83,
  AED: 0.044,  SGD: 0.016,  THB: 0.43,  MYR: 0.057,  AUD: 0.018,
  CAD: 0.016,  CHF: 0.011,  CNY: 0.087, NPR: 1.6,    LKR: 3.6,
  IDR: 195.0,  PHP: 0.69,   VND: 305.0, TRY: 0.46,   SAR: 0.045,
  KRW: 16.5,   HKD: 0.094,  NZD: 0.020, ZAR: 0.22,   BRL: 0.069,
  RUB: 1.10,
};

const SYMBOLS = {
  INR: "₹", USD: "$", EUR: "€", GBP: "£", JPY: "¥", AED: "د.إ",
  SGD: "S$", THB: "฿", MYR: "RM", AUD: "A$", CAD: "C$", CHF: "Fr",
  CNY: "¥", NPR: "रू", LKR: "Rs", IDR: "Rp", PHP: "₱", VND: "₫",
  TRY: "₺", SAR: "ر.س", KRW: "₩", HKD: "HK$", NZD: "NZ$",
  ZAR: "R", BRL: "R$", RUB: "₽",
};

export const onRequestOptions = () => jsonResponse({ ok: true });

export const onRequestGet = async ({ request }) => {
  const url = new URL(request.url);
  const from = (url.searchParams.get("from") || "INR").toUpperCase();
  const to = (url.searchParams.get("to") || "USD").toUpperCase();
  const amount = parseFloat(url.searchParams.get("amount") || "1000") || 1000;

  // Try live rates with a tight timeout so we never block the page.
  let rate = null;
  let source = "fallback";
  try {
    const ctrl = new AbortController();
    const tm = setTimeout(() => ctrl.abort(), 4000);
    const r = await fetch(`https://api.exchangerate.host/convert?from=${from}&to=${to}&amount=1`, { signal: ctrl.signal });
    clearTimeout(tm);
    if (r.ok) {
      const j = await r.json();
      if (j?.success && j?.result) { rate = j.result; source = "exchangerate.host"; }
    }
  } catch (_) { /* fall through to fallback */ }

  if (rate === null) {
    const fromRate = FALLBACK_RATES[from] ?? 1;
    const toRate = FALLBACK_RATES[to] ?? 1;
    rate = toRate / fromRate;
  }

  const converted = +(amount * rate).toFixed(2);
  return jsonResponse({
    ok: true,
    from, to, amount, rate: +rate.toFixed(6),
    converted,
    formatted: `${SYMBOLS[to] || to} ${converted.toLocaleString("en-IN", { maximumFractionDigits: 2 })}`,
    source,
    supported: Object.keys(FALLBACK_RATES),
    timestamp: new Date().toISOString(),
  });
};
