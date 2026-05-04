/* ════════════════════════════════════════════════════════════════
   _shared/wiki.js — Wikipedia photo + extract enrichment.

   Edge-safe: uses fetch() against the public REST API. Returns
   { thumbnail, image, extract, url } for a given title. Falls back
   gracefully on network or 404 errors (returns null fields, never
   throws). Includes a small in-memory LRU cache so repeat lookups
   inside the same isolate are free.
   ════════════════════════════════════════════════════════════════ */

const CACHE = new Map();
const CACHE_MAX = 500;

function cacheGet(k) {
  if (!CACHE.has(k)) return null;
  const v = CACHE.get(k);
  CACHE.delete(k); CACHE.set(k, v); // bump LRU
  return v;
}
function cacheSet(k, v) {
  if (CACHE.size >= CACHE_MAX) {
    const first = CACHE.keys().next().value;
    if (first) CACHE.delete(first);
  }
  CACHE.set(k, v);
}

const EMPTY = { thumbnail: null, image: null, extract: null, url: null, ok: false };

export async function fetchWikiSummary(title) {
  if (!title) return EMPTY;
  const key = String(title).trim();
  if (!key) return EMPTY;
  const cached = cacheGet(key);
  if (cached) return cached;

  // Wikipedia REST: replace spaces with underscores, encode the rest
  const enc = encodeURIComponent(key.replace(/\s+/g, "_"));
  const url = `https://en.wikipedia.org/api/rest_v1/page/summary/${enc}`;

  try {
    const ctrl = new AbortController();
    const tm = setTimeout(() => ctrl.abort(), 4500);
    const r = await fetch(url, {
      headers: { "User-Agent": "SmartRouteSRMIST/6.1 (https://smartroute-srmist.pages.dev)" },
      signal: ctrl.signal,
    });
    clearTimeout(tm);
    if (!r.ok) { cacheSet(key, EMPTY); return EMPTY; }
    const j = await r.json();
    const out = {
      thumbnail: j.thumbnail?.source || null,
      image:     j.originalimage?.source || j.thumbnail?.source || null,
      extract:   j.extract || null,
      url:       j.content_urls?.desktop?.page || `https://en.wikipedia.org/wiki/${enc}`,
      ok:        true,
    };
    cacheSet(key, out);
    return out;
  } catch {
    cacheSet(key, EMPTY);
    return EMPTY;
  }
}

/* Enrich many items in parallel, with per-item fall-back to a
   "search the place name" thumbnail via Wikipedia's prefix search. */
export async function enrichWithPhotos(items, opts = {}) {
  const {
    titleKey = "wikiTitle",
    fallbackKey = "name",
    cityForFallback = "",
    concurrency = 8,
  } = opts;

  if (!Array.isArray(items) || items.length === 0) return items || [];

  // Build a unique work-list (so duplicates don't double-fetch)
  const work = [];
  for (const it of items) {
    const t = it?.[titleKey] || it?.[fallbackKey];
    if (t) work.push({ item: it, title: t });
  }

  // Bounded concurrency
  let inFlight = 0, idx = 0;
  return await new Promise((resolve) => {
    const out = [...items];
    let pending = work.length;
    if (pending === 0) return resolve(out);

    const next = () => {
      while (inFlight < concurrency && idx < work.length) {
        const cur = work[idx++];
        inFlight++;
        (async () => {
          let r = await fetchWikiSummary(cur.title);
          // Fallback: try "<name> <city>"
          if (!r.ok && cityForFallback) {
            r = await fetchWikiSummary(`${cur.title} ${cityForFallback}`);
          }
          if (r.ok) {
            cur.item.thumbnail = cur.item.thumbnail || r.thumbnail;
            cur.item.image     = cur.item.image     || r.image;
            cur.item.extract   = cur.item.extract   || r.extract;
            cur.item.wikiUrl   = cur.item.wikiUrl   || r.url;
          } else if (!cur.item.thumbnail) {
            // Last-resort placeholder via Unsplash source (free, no key)
            const q = encodeURIComponent((cur.title || cur.item.name || "travel") + (cityForFallback ? " " + cityForFallback : ""));
            cur.item.thumbnail = cur.item.thumbnail || `https://source.unsplash.com/600x400/?${q}`;
            cur.item.image     = cur.item.image     || `https://source.unsplash.com/1600x900/?${q}`;
            cur.item.imageFallback = true;
          }
          inFlight--; pending--;
          if (pending === 0) resolve(out);
          else next();
        })().catch(() => {
          inFlight--; pending--;
          if (pending === 0) resolve(out);
          else next();
        });
      }
    };
    next();
  });
}

/* Quick helper: synchronously build a deterministic Unsplash fallback
   for any place name (used when we can't afford another network round-trip,
   e.g. for restaurants in the itinerary response). */
export function unsplashFor(query, size = "600x400") {
  const q = encodeURIComponent(query || "travel");
  return `https://source.unsplash.com/${size}/?${q}`;
}
