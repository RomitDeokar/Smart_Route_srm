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
            // Last-resort: deterministic LoremFlickr (real travel photos, free, no key)
            const q = (cur.title || cur.item.name || "travel") + (cityForFallback ? " " + cityForFallback : "");
            cur.item.thumbnail = cur.item.thumbnail || flickrFor(q, "600x400");
            cur.item.image     = cur.item.image     || flickrFor(q, "1600x900");
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

/* ── Image fallback chain ──────────────────────────────────────────
   source.unsplash.com is deprecated (HTTP 503 since 2024). We use:

   1. LoremFlickr      — real Flickr travel photos, deterministic via tag
   2. Picsum (seeded)  — works offline, but generic landscapes
   3. Curated banks    — hand-picked Wikimedia Commons URLs per category

   These are all free, keyless, CDN-cached and embed-safe.
   ────────────────────────────────────────────────────────────────── */

// Curated Wikimedia Commons fallbacks by keyword category.
const COMMONS_BANK = {
  beach:      "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fc/BeachFun.jpg/1280px-BeachFun.jpg",
  mountain:   "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Everest_kalapatthar.jpg/1280px-Everest_kalapatthar.jpg",
  temple:     "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/Hindu_temple_-_Madurai.jpg/1280px-Hindu_temple_-_Madurai.jpg",
  fort:       "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/Amber_Fort_Jaipur_2.jpg/1280px-Amber_Fort_Jaipur_2.jpg",
  palace:     "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dc/Hawa_Mahal_2011.jpg/1280px-Hawa_Mahal_2011.jpg",
  city:       "https://upload.wikimedia.org/wikipedia/commons/thumb/3/35/Mumbai_skyline_at_night.jpg/1280px-Mumbai_skyline_at_night.jpg",
  food:       "https://upload.wikimedia.org/wikipedia/commons/thumb/6/65/Indian_thali_meal_-_Bangalore.jpg/1280px-Indian_thali_meal_-_Bangalore.jpg",
  restaurant: "https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Indian_thali_at_a_restaurant.jpg/1280px-Indian_thali_at_a_restaurant.jpg",
  hotel:      "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Taj_Mahal_Palace_Hotel.jpg/1280px-Taj_Mahal_Palace_Hotel.jpg",
  museum:     "https://upload.wikimedia.org/wikipedia/commons/thumb/0/06/Indian_Museum_Kolkata_Front.jpg/1280px-Indian_Museum_Kolkata_Front.jpg",
  park:       "https://upload.wikimedia.org/wikipedia/commons/thumb/3/35/Lalbagh_glasshouse_Bangalore.jpg/1280px-Lalbagh_glasshouse_Bangalore.jpg",
  default:    "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cf/India_Gate_New_Delhi.jpg/1280px-India_Gate_New_Delhi.jpg",
};

function pickBankImage(query) {
  const q = String(query || "").toLowerCase();
  for (const k of Object.keys(COMMONS_BANK)) {
    if (k !== "default" && q.includes(k)) return COMMONS_BANK[k];
  }
  return COMMONS_BANK.default;
}

/* Build a deterministic LoremFlickr URL from any query string. Returns a
   real photo URL that works without an API key. Size defaults to 600x400.
   Accepts "WIDTHxHEIGHT" string sizes for backwards compatibility. */
export function flickrFor(query, size = "600x400") {
  const [w = 600, h = 400] = String(size).split("x").map((n) => parseInt(n, 10));
  // Pick 3 best tags from the query for relevance
  const tags = String(query || "travel landmark")
    .toLowerCase()
    .replace(/[^a-z0-9 ]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 2)
    .slice(0, 3)
    .join(",");
  // lock=N gives a deterministic photo for a given tag-set
  const lock = Math.abs(hashCode(query || "travel")) % 1000;
  return `https://loremflickr.com/${w}/${h}/${encodeURIComponent(tags || "travel")}?lock=${lock}`;
}

function hashCode(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = ((h << 5) - h + s.charCodeAt(i)) | 0;
  return h;
}

/* Backwards-compatible alias kept (lots of call-sites use unsplashFor).
   Now points to the working flickrFor pipeline. */
export function unsplashFor(query, size = "600x400") {
  return flickrFor(query, size);
}

/* Curated Wikimedia bank — used as a 2nd-tier fallback when even
   LoremFlickr is rate-limited. Keyless, always-on. */
export function commonsBankFor(query) {
  return pickBankImage(query);
}
