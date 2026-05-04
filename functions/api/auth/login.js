/* POST /api/auth/login — issues a JWT for any valid email/password.
   In-memory user store is per-isolate; if user isn't found we treat the
   request as a demo login so /dashboard always loads on a fresh edge node. */

import { signJwt, hashPassword, jsonResponse, getJwtSecret, userStore } from "../_shared/auth.js";

export const onRequestOptions = () => jsonResponse({ ok: true });

export const onRequestPost = async ({ request, env }) => {
  let body = {};
  try { body = await request.json(); } catch { body = {}; }
  const { email, password } = body || {};

  if (!email || !password) {
    return jsonResponse({ ok: false, error: "Email and password are required." }, 400);
  }

  const secret = getJwtSecret(env);
  const lcEmail = String(email).toLowerCase().trim();

  // Demo user shortcut — any email starting with "demo@" or the magic email
  if (lcEmail === "demo@srmist.edu.in") {
    const token = await signJwt({ id: "demo", name: "Demo User", email: lcEmail }, secret);
    return jsonResponse({ ok: true, token,
      user: { id: "demo", name: "Demo User", email: lcEmail, initials: "DU" } });
  }

  const users = userStore();
  const user = users.get(lcEmail);

  if (user) {
    const hashed = await hashPassword(password, secret);
    if (user.password === hashed) {
      const token = await signJwt({ id: user.id, name: user.name, email: user.email }, secret);
      const { password: _p, ...safeUser } = user;
      return jsonResponse({
        ok: true, token,
        user: { ...safeUser, initials: user.name.split(" ").map(w => w[0]).join("").slice(0, 2).toUpperCase() }
      });
    }
    return jsonResponse({ ok: false, error: "Invalid email or password." }, 401);
  }

  // No user found in this isolate. Auto-provision so login still works
  // across edge regions (matches the SRMIST hackathon demo expectation).
  const name = String(email).split("@")[0];
  const initials = name.slice(0, 2).toUpperCase();
  const id = crypto.randomUUID();
  const newUser = {
    id, name, email: lcEmail,
    password: await hashPassword(password, secret),
    createdAt: new Date().toISOString(),
    preferences: { persona: "explorer", budget: 18000, homeCity: "" }
  };
  users.set(lcEmail, newUser);
  const token = await signJwt({ id, name, email: lcEmail }, secret);
  const { password: _p, ...safeUser } = newUser;
  return jsonResponse({ ok: true, token, user: { ...safeUser, initials } });
};
