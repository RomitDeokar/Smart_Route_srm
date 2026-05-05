/* ════════════════════════════════════════════════════════════════
   functions/api/_shared/auth.js — JWT helpers for Cloudflare Workers
   Web Crypto only (no Node.js Buffer). HS256 sign + verify.
   ════════════════════════════════════════════════════════════════ */

const enc = new TextEncoder();
const dec = new TextDecoder();

function b64urlFromBytes(bytes) {
  let s = "";
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}
function b64urlFromString(str) {
  return b64urlFromBytes(enc.encode(str));
}
function b64urlToBytes(s) {
  s = s.replace(/-/g, "+").replace(/_/g, "/");
  while (s.length % 4) s += "=";
  const bin = atob(s);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

async function getKey(secret) {
  return crypto.subtle.importKey(
    "raw", enc.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false, ["sign", "verify"]
  );
}

export async function signJwt(payload, secret) {
  const header = { alg: "HS256", typ: "JWT" };
  const now = Math.floor(Date.now() / 1000);
  const body = { ...payload, iat: now, exp: now + 7 * 24 * 3600 };
  const h = b64urlFromString(JSON.stringify(header));
  const b = b64urlFromString(JSON.stringify(body));
  const key = await getKey(secret);
  const sigBuf = await crypto.subtle.sign("HMAC", key, enc.encode(`${h}.${b}`));
  const sig = b64urlFromBytes(new Uint8Array(sigBuf));
  return `${h}.${b}.${sig}`;
}

export async function verifyJwt(token, secret) {
  try {
    const [h, b, sig] = String(token || "").split(".");
    if (!h || !b || !sig) return null;
    const key = await getKey(secret);
    const sigBytes = b64urlToBytes(sig);
    const ok = await crypto.subtle.verify("HMAC", key, sigBytes, enc.encode(`${h}.${b}`));
    if (!ok) return null;
    const payload = JSON.parse(dec.decode(b64urlToBytes(b)));
    if (payload.exp && payload.exp < Math.floor(Date.now() / 1000)) return null;
    return payload;
  } catch {
    return null;
  }
}

export async function hashPassword(password, secret) {
  const buf = await crypto.subtle.digest(
    "SHA-256",
    enc.encode(String(password) + ":" + secret)
  );
  const arr = new Uint8Array(buf);
  return Array.from(arr).map(b => b.toString(16).padStart(2, "0")).join("");
}

export function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Headers": "Content-Type, Authorization",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS"
    }
  });
}

export function getJwtSecret(env) {
  return (env && env.JWT_SECRET) || "smartroute-srmist-cf-secret-2026";
}

/* ── Tiny in-memory user store. Workers reset across invocations,
   so this is per-isolate only. The Login.jsx / Register.jsx code
   transparently falls back to localStorage demo mode if a fresh
   isolate doesn't recognise the credentials, so the UX still works. */
const __users = new Map();
export function userStore() { return __users; }
