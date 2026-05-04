/* ════════════════════════════════════════════════════════════════
   src/lib/api.js — robust API client used across the app.
   Wraps `fetch` so that empty bodies / non-JSON responses don't
   throw the dreaded:
     "Failed to execute 'json' on 'Response': Unexpected end of JSON input"
   ════════════════════════════════════════════════════════════════ */

/* Safely parse a response — never throws. Returns:
   { ok: <bool>, status: <int>, data: <any|null>, error: <string|null>, raw: <string> } */
export async function safeJson(res) {
  let raw = "";
  try { raw = await res.text(); } catch { raw = ""; }

  let data = null;
  if (raw) {
    try { data = JSON.parse(raw); }
    catch { data = null; }
  }

  if (!res.ok) {
    return {
      ok: false,
      status: res.status,
      data,
      error: (data && data.error) ||
             `Request failed (HTTP ${res.status})`,
      raw
    };
  }

  if (data === null) {
    // Empty body or non-JSON — surface as a soft error so callers can
    // fall back to a mock instead of crashing.
    return { ok: false, status: res.status, data: null,
             error: "Empty or non-JSON response from server", raw };
  }

  return { ok: data.ok !== false, status: res.status, data,
           error: data.error || null, raw };
}

/* High-level wrapper: api(url, init?) → { ok, data, error, status } */
export async function api(url, init = {}) {
  const headers = {
    "Content-Type": "application/json",
    ...(init.headers || {})
  };
  const token = localStorage.getItem("sr_token");
  if (token && !headers.Authorization) headers.Authorization = `Bearer ${token}`;

  try {
    const res = await fetch(url, { ...init, headers });
    return await safeJson(res);
  } catch (networkErr) {
    return {
      ok: false,
      status: 0,
      data: null,
      error: networkErr?.message || "Network error",
      raw: ""
    };
  }
}

export default api;
