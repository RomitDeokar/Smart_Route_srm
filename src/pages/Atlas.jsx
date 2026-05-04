import { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";

/* Travel Atlas — visualizes all visited/planned trips on a Leaflet map.
   Stores trips in localStorage under key "sr_atlas_trips". */

const ATLAS_KEY = "sr_atlas_trips";

function ensureLeaflet() {
  return new Promise((resolve) => {
    if (window.L) { resolve(window.L); return; }
    if (!document.getElementById("leaflet-css")) {
      const css = document.createElement("link");
      css.id = "leaflet-css"; css.rel = "stylesheet";
      css.href = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css";
      document.head.appendChild(css);
    }
    if (!document.getElementById("leaflet-js")) {
      const js = document.createElement("script");
      js.id = "leaflet-js";
      js.src = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js";
      js.onload = () => resolve(window.L);
      document.head.appendChild(js);
    } else {
      const wait = setInterval(() => {
        if (window.L) { clearInterval(wait); resolve(window.L); }
      }, 100);
    }
  });
}

const SAMPLE_TRIPS = [
  { destination: "Goa",       lat: 15.2993, lon: 74.124,  days: 4, budget: 22000, country: "India", continent: "Asia", date: "2025-12-10" },
  { destination: "Jaipur",    lat: 26.9124, lon: 75.7873, days: 3, budget: 15000, country: "India", continent: "Asia", date: "2025-09-22" },
  { destination: "Manali",    lat: 32.2396, lon: 77.1887, days: 5, budget: 28000, country: "India", continent: "Asia", date: "2025-06-15" },
  { destination: "Kerala",    lat: 9.9312,  lon: 76.2673, days: 6, budget: 35000, country: "India", continent: "Asia", date: "2024-11-08" },
];

export default function Atlas({ tripCtx, addToast }) {
  const [trips, setTrips] = useState(() => {
    try {
      const raw = localStorage.getItem(ATLAS_KEY);
      const parsed = raw ? JSON.parse(raw) : null;
      return Array.isArray(parsed) && parsed.length ? parsed : SAMPLE_TRIPS;
    } catch { return SAMPLE_TRIPS; }
  });
  const mapRef = useRef(null);
  const mapObj = useRef(null);
  const layersRef = useRef([]);

  useEffect(() => {
    try { localStorage.setItem(ATLAS_KEY, JSON.stringify(trips)); } catch {}
  }, [trips]);

  // Render the Leaflet atlas map
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const L = await ensureLeaflet();
      if (cancelled || !mapRef.current) return;
      if (mapObj.current) { mapObj.current.remove(); mapObj.current = null; }
      const map = L.map(mapRef.current, { center: [20.5937, 78.9629], zoom: 4, attributionControl: false, preferCanvas: true });
      mapObj.current = map;
      L.tileLayer("https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
        { maxZoom: 19, subdomains: "abcd", detectRetina: true }).addTo(map);
      drawTrips(L);
      [80, 240, 600, 1200].forEach(t => setTimeout(() => mapObj.current && mapObj.current.invalidateSize(true), t));
    })();
    const onResize = () => mapObj.current && mapObj.current.invalidateSize(true);
    window.addEventListener("resize", onResize);
    return () => {
      cancelled = true;
      window.removeEventListener("resize", onResize);
      if (mapObj.current) { mapObj.current.remove(); mapObj.current = null; }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (mapObj.current && window.L) drawTrips(window.L);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trips]);

  function drawTrips(L) {
    const map = mapObj.current; if (!map) return;
    layersRef.current.forEach(l => map.removeLayer(l));
    layersRef.current = [];
    const pts = [];
    trips.forEach((t, i) => {
      if (typeof t.lat !== "number" || typeof t.lon !== "number") return;
      const c = L.circleMarker([t.lat, t.lon], {
        radius: 9, color: "#2563eb", fillColor: "#2563eb", fillOpacity: 0.7, weight: 2,
      })
      .bindPopup(`<div style="font-family:'Plus Jakarta Sans',sans-serif">
        <div style="font-weight:700;font-size:13px;color:#111827">${t.destination}</div>
        <div style="font-size:11.5px;color:#6b7280;margin-top:3px">${t.date || ""} · ${t.days || 1} day${t.days===1?'':'s'}</div>
        <div style="font-size:11.5px;color:#16a34a;font-weight:700">₹${(t.budget||0).toLocaleString("en-IN")}</div>
      </div>`)
      .addTo(map);
      layersRef.current.push(c);
      pts.push([t.lat, t.lon]);
    });
    if (pts.length > 1) {
      const poly = L.polyline(pts, { color: "#7c3aed", weight: 2, opacity: 0.4, dashArray: "6 6" }).addTo(map);
      layersRef.current.push(poly);
      try { map.fitBounds(L.latLngBounds(pts), { padding: [40, 40] }); } catch {}
    } else if (pts.length === 1) {
      map.setView(pts[0], 6);
    }
  }

  const addCurrent = () => {
    if (!tripCtx?.destination) { addToast("No destination to add", "warning"); return; }
    const dest = tripCtx.destination;
    const exists = trips.find(t => t.destination?.toLowerCase() === dest.toLowerCase());
    if (exists) { addToast("Already in atlas", "info"); return; }
    // Lookup approximate coords (synced with Dashboard)
    const COORDS = {
      goa:[15.2993,74.124], delhi:[28.6139,77.209], mumbai:[19.076,72.8777],
      chennai:[13.0827,80.2707], bangalore:[12.9716,77.5946], jaipur:[26.9124,75.7873],
      kolkata:[22.5726,88.3639], hyderabad:[17.385,78.4867], manali:[32.2396,77.1887],
      shimla:[31.1048,77.1734], kerala:[9.9312,76.2673], kochi:[9.9312,76.2673],
      udaipur:[24.5854,73.7125], agra:[27.1767,78.0081], varanasi:[25.3176,83.0064],
      ooty:[11.4102,76.695], rishikesh:[30.0869,78.2676],
    };
    const k = dest.toLowerCase().split(",")[0].trim();
    let coords = COORDS[k];
    if (!coords) {
      for (const [c, co] of Object.entries(COORDS)) if (k.includes(c) || c.includes(k)) { coords = co; break; }
    }
    if (!coords) coords = [20.5937, 78.9629];
    const newTrip = {
      destination: dest, lat: coords[0], lon: coords[1],
      days: tripCtx.days || 3, budget: tripCtx.budget || 0,
      country: "India", continent: "Asia",
      date: new Date().toISOString().split("T")[0],
    };
    setTrips([newTrip, ...trips]);
    addToast(`Added ${dest} to your atlas`, "success");
  };

  const removeTrip = (idx) => {
    setTrips(trips.filter((_, i) => i !== idx));
    addToast("Removed from atlas", "info");
  };

  const clearAtlas = () => {
    if (!confirm("Clear all trips from your atlas?")) return;
    setTrips([]); addToast("Atlas cleared", "info");
  };

  // Stats
  const countries = new Set(trips.map(t => t.country).filter(Boolean));
  const continents = new Set(trips.map(t => t.continent).filter(Boolean));
  const totalBudget = trips.reduce((s, t) => s + (t.budget || 0), 0);
  const totalDays = trips.reduce((s, t) => s + (t.days || 0), 0);
  const totalDistance = (() => {
    if (trips.length < 2) return 0;
    let d = 0;
    for (let i = 1; i < trips.length; i++) {
      const a = trips[i-1], b = trips[i];
      if (typeof a.lat !== "number" || typeof b.lat !== "number") continue;
      const R = 6371, toRad = x => x * Math.PI / 180;
      const dLat = toRad(b.lat - a.lat), dLon = toRad(b.lon - a.lon);
      const x = Math.sin(dLat/2)**2 + Math.cos(toRad(a.lat))*Math.cos(toRad(b.lat))*Math.sin(dLon/2)**2;
      d += 2 * R * Math.asin(Math.sqrt(x));
    }
    return Math.round(d);
  })();

  const STATS = [
    { label: "Trips logged",     value: trips.length, color: "var(--blue)" },
    { label: "Countries visited",value: countries.size, color: "var(--green)" },
    { label: "Continents",       value: continents.size, color: "var(--purple)" },
    { label: "Total distance",   value: `${totalDistance.toLocaleString("en-IN")} km`, color: "var(--amber)" },
    { label: "Total days",       value: totalDays, color: "var(--teal)" },
    { label: "Total spend",      value: `₹${totalBudget.toLocaleString("en-IN")}`, color: "#db2777" },
  ];

  return (
    <div>
      <div className="section-title">Travel Atlas</div>
      <div className="section-sub">Your visual journey across destinations · auto-saved to local storage</div>

      <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fit,minmax(140px,1fr))", gap:12, marginBottom:18 }}>
        {STATS.map((s,i)=>(
          <motion.div key={s.label} initial={{opacity:0,y:6}} animate={{opacity:1,y:0}} transition={{delay:i*0.05}}
            className="stat-card" style={{ borderLeft:`3px solid ${s.color}` }}>
            <div className="stat-label">{s.label}</div>
            <div className="stat-value" style={{ color: s.color, fontSize:20 }}>{s.value}</div>
          </motion.div>
        ))}
      </div>

      <div className="card" style={{ marginBottom:18 }}>
        <div className="card-header">
          <span className="card-title">🗺️ World Map · {trips.length} pinned</span>
          <div style={{ display:"flex", gap:6 }}>
            <button className="btn btn-primary btn-sm" onClick={addCurrent}>+ Add current trip</button>
            <button className="btn btn-ghost btn-sm" onClick={clearAtlas}>Clear</button>
          </div>
        </div>
        <div style={{ padding:0 }}>
          <div ref={mapRef} style={{ width:"100%", height:"460px" }} />
        </div>
      </div>

      <div className="card">
        <div className="card-header"><span className="card-title">Trip log ({trips.length})</span></div>
        <div className="card-body">
          {trips.length === 0 ? (
            <div style={{ color:"var(--text-3)", fontSize:13, textAlign:"center", padding:"20px 0" }}>
              No trips yet. Plan one in the Itinerary tab and tap "+ Add current trip".
            </div>
          ) : (
            <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fill,minmax(220px,1fr))", gap:10 }}>
              {trips.map((t,i)=>(
                <div key={i} style={{ padding:"12px 14px", border:"1.5px solid var(--border)", borderRadius:"var(--r-md)", background:"var(--bg-soft)" }}>
                  <div style={{ fontSize:14, fontWeight:700, color:"var(--text)" }}>{t.destination}</div>
                  <div style={{ fontSize:11.5, color:"var(--text-3)", marginTop:3 }}>{t.country || "—"} · {t.date || ""}</div>
                  <div style={{ fontSize:12, color:"var(--text-2)", marginTop:5 }}>{t.days} day{t.days===1?'':'s'} · ₹{(t.budget||0).toLocaleString("en-IN")}</div>
                  <button onClick={()=>removeTrip(i)} style={{ marginTop:6, fontSize:11, color:"var(--red)", background:"transparent", border:"none", cursor:"pointer", padding:0, fontWeight:600 }}>Remove</button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
