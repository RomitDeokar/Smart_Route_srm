/* GET /api/auth/me — returns current user from Bearer token. */

import { verifyJwt, jsonResponse, getJwtSecret, userStore } from "../_shared/auth.js";

export const onRequestOptions = () => jsonResponse({ ok: true });

export const onRequestGet = async ({ request, env }) => {
  const auth = request.headers.get("Authorization") || "";
  const token = auth.startsWith("Bearer ") ? auth.slice(7) : null;
  if (!token) return jsonResponse({ ok: false, error: "Authentication required." }, 401);

  const secret = getJwtSecret(env);
  const payload = await verifyJwt(token, secret);
  if (!payload) return jsonResponse({ ok: false, error: "Invalid or expired token." }, 401);

  const user = userStore().get(payload.email) || {
    id: payload.id, name: payload.name, email: payload.email
  };
  const { password: _p, ...safe } = user;
  return jsonResponse({ ok: true, user: safe });
};
