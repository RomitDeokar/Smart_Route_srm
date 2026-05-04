/* POST /api/autonomous/scout — Autonomous scout that surfaces hidden gems
   and persona-aligned attractions for a destination. */

import { jsonResponse } from "../_shared/auth.js";
import { getTopAttractions, geocode } from "../_shared/cities.js";

export const onRequestOptions = () => jsonResponse({ ok: true });

const PERSONA_FILTER = {
  explorer:  ["heritage","viewpoint","nature","temple"],
  student:   ["market","food","viewpoint","beach"],
  family:    ["temple","park","beach","museum"],
  creator:   ["viewpoint","beach","heritage","architecture"],
  luxury:    ["heritage","palace","fort","fine-dining"],
  adventure: ["nature","viewpoint","trek","activity"],
};

export const onRequestPost = async ({ request }) => {
  let body = {};
  try { body = await request.json(); } catch { body = {}; }
  const { destination, persona = "explorer" } = body;

  if (!destination) return jsonResponse({ ok: false, error: "destination required" }, 400);

  const geo = geocode(destination);
  const all = getTopAttractions(geo.resolvedCity || destination.toLowerCase().split(",")[0].trim());

  const filter = PERSONA_FILTER[persona] || PERSONA_FILTER.explorer;
  const ranked = all.map(a => {
    const score = filter.reduce((acc, f) =>
      acc + ((a.type || "").toLowerCase().includes(f) || (a.description || "").toLowerCase().includes(f) ? 0.25 : 0), 0.4);
    return { ...a, score: Math.min(1, score) };
  }).sort((a, b) => b.score - a.score);

  // Hidden gems = beyond top-3
  const top = ranked.slice(0, 3);
  const hiddenGems = ranked.slice(3, 9);

  return jsonResponse({
    ok: true,
    mode: "autonomous-scout",
    destination,
    persona,
    summary: `Scout surfaced ${ranked.length} candidates · top ${top.length} highlights · ${hiddenGems.length} hidden gems`,
    top,
    hiddenGems,
    method: "Persona-weighted scoring with type/description matching",
  });
};
