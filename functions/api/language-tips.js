/* POST /api/language-tips — local-language phrases per city. */

import { jsonResponse } from "./_shared/auth.js";
import { getLanguageTips } from "./_shared/extras.js";

export const onRequestOptions = () => jsonResponse({ ok: true });

export const onRequestPost = async ({ request }) => {
  let body = {};
  try { body = await request.json(); } catch { body = {}; }
  const { destination } = body || {};
  return jsonResponse({ ok: true, ...getLanguageTips(destination || "") });
};
