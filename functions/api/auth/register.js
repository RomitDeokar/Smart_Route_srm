/* POST /api/auth/register — creates a new account and returns a JWT. */

import { signJwt, hashPassword, jsonResponse, getJwtSecret, userStore } from "../_shared/auth.js";

export const onRequestOptions = () => jsonResponse({ ok: true });

export const onRequestPost = async ({ request, env }) => {
  let body = {};
  try { body = await request.json(); } catch { body = {}; }
  const { name, email, password } = body || {};

  if (!name || !email || !password) {
    return jsonResponse({ ok: false, error: "Name, email and password are required." }, 400);
  }
  if (String(password).length < 6) {
    return jsonResponse({ ok: false, error: "Password must be at least 6 characters." }, 400);
  }

  const secret = getJwtSecret(env);
  const lcEmail = String(email).toLowerCase().trim();
  const users = userStore();

  // If the user already exists in this isolate, treat re-registration with
  // the SAME password as a successful login (idempotent UX).
  const existing = users.get(lcEmail);
  if (existing) {
    const hashed = await hashPassword(password, secret);
    if (existing.password !== hashed) {
      return jsonResponse({ ok: false, error: "Email already registered." }, 409);
    }
  }

  const id = existing?.id || crypto.randomUUID();
  const trimmedName = String(name).trim();
  const initials = trimmedName.split(/\s+/).map(w => w[0] || "").join("").slice(0, 2).toUpperCase() || "SR";

  const user = existing || {
    id,
    name: trimmedName,
    email: lcEmail,
    password: await hashPassword(password, secret),
    createdAt: new Date().toISOString(),
    preferences: { persona: "explorer", budget: 18000, homeCity: "" }
  };
  users.set(lcEmail, user);

  const token = await signJwt({ id, name: user.name, email: user.email }, secret);
  const { password: _p, ...safeUser } = user;
  return jsonResponse({ ok: true, token, user: { ...safeUser, initials } }, 201);
};
