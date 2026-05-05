/* GET /api/health — server status. */
import { jsonResponse } from "./_shared/auth.js";

export const onRequestOptions = () => jsonResponse({ ok: true });
export const onRequestGet = () => jsonResponse({
  ok: true,
  version: "5.1.0",
  provider: "cloudflare-pages-functions",
  timestamp: new Date().toISOString(),
});
