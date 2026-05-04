import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import CrowdAnalyzer from "../components/CrowdAnalyzer.jsx";
import LanguageTips from "../components/LanguageTips.jsx";
import TripChecklist from "../components/TripChecklist.jsx";
import ItineraryMap from "../components/ItineraryMap.jsx";

// Deterministic LoremFlickr fallback (source.unsplash.com is deprecated, returns 503).
function flickrUrl(query, w = 600, h = 400) {
  const tags = String(query || "travel landmark")
    .toLowerCase().replace(/[^a-z0-9 ]/g, " ")
    .split(/\s+/).filter(t => t.length > 2).slice(0, 3).join(",");
  let h1 = 0; const s = String(query || "travel");
  for (let i = 0; i < s.length; i++) h1 = ((h1 << 5) - h1 + s.charCodeAt(i)) | 0;
  const lock = Math.abs(h1) % 1000;
  return `https://loremflickr.com/${w}/${h}/${encodeURIComponent(tags || "travel")}?lock=${lock}`;
}

const PERSONAS = [
  { id:"explorer",  label:"Explorer",   sub:"Hidden gems" },
  { id:"student",   label:"Student",    sub:"Budget travel" },
  { id:"family",    label:"Family",     sub:"Comfort first" },
  { id:"creator",   label:"Creator",    sub:"Visual stories" },
  { id:"luxury",    label:"Luxury",     sub:"Premium stays" },
  { id:"adventure", label:"Adventure",  sub:"Off-beat" },
];
const SERVICES = ["Hotels","Food","Cab rental","Attractions","Language tips","Local events","Rain backup"];
const STOP_COLORS = {
  Attraction: { bg:"var(--blue-dim)",        color:"var(--blue)" },
  Restaurant: { bg:"var(--green-bg)",        color:"var(--green)" },
  Activity:   { bg:"var(--purple-bg)",       color:"var(--purple)" },
  Shopping:   { bg:"rgba(244,114,182,0.10)", color:"#db2777" },
  Dinner:     { bg:"var(--amber-bg)",        color:"var(--amber)" },
};

export default function Itinerary({ tripCtx, addToast }) {
  const [form, setForm]           = useState({ ...tripCtx });
  const [itinerary, setItin]      = useState(null);
  const [loading, setLoading]     = useState(false);
  const [activeDay, setActiveDay] = useState(1);
  const [activeTab, setActiveTab] = useState("plan");
  const [mapAllDays, setMapAll]   = useState(true);

  // ── Agentic AI automation state ──────────────────────────────────────
  const [autoMode, setAutoMode]   = useState(true);   // auto-execute pipeline + self-heal
  const [agentLog, setAgentLog]   = useState([]);     // live agent activity feed
  const [aiStatus, setAiStatus]   = useState(null);   // /api/autonomous/status snapshot
  const [confidence, setConf]     = useState(null);   // last plan confidence
  const [healing, setHealing]     = useState(false);  // self-heal in progress
  const [pipelineStage, setStage] = useState(0);      // current active stage 0..N
  const [taskQueue, setTaskQueue] = useState([]);     // parallel agentic tasks
  const heartbeatRef              = useRef(null);

  // Pipeline stages used for the visual progress bar
  const PIPELINE = [
    { id:"scout",   label:"Scout",         icon:"🔍" },
    { id:"geo",     label:"Geocode",       icon:"📍" },
    { id:"poi",     label:"POI Filter",    icon:"🗺️" },
    { id:"weather", label:"Weather AI",    icon:"☁️" },
    { id:"crowd",   label:"Crowd GP",      icon:"👥" },
    { id:"budget",  label:"Budget Q",      icon:"💸" },
    { id:"route",   label:"MCTS Route",    icon:"🧭" },
    { id:"flights", label:"Flights",       icon:"✈️" },
    { id:"trains",  label:"Trains",        icon:"🚆" },
    { id:"hotels",  label:"Hotels",        icon:"🏨" },
    { id:"cabs",    label:"Cabs",          icon:"🚖" },
    { id:"critic",  label:"Self-Critic",   icon:"🧠" },
  ];

  useEffect(() => { setForm(f => ({ ...f, ...tripCtx })); }, [tripCtx]);

  const pushLog = (msg, kind="info") => {
    setAgentLog(l => [{ ts: Date.now(), msg, kind }, ...l].slice(0, 40));
  };

  // Live agent fleet heartbeat (polls /api/autonomous/status every 8s when autoMode is on)
  useEffect(() => {
    if (!autoMode) { if (heartbeatRef.current) { clearInterval(heartbeatRef.current); heartbeatRef.current = null; } return; }
    let mounted = true;
    const poll = async () => {
      try {
        const r = await fetch("/api/autonomous/status");
        const j = await r.json();
        if (mounted && j) setAiStatus(j);
      } catch {}
    };
    poll();
    heartbeatRef.current = setInterval(poll, 8000);
    return () => { mounted = false; if (heartbeatRef.current) clearInterval(heartbeatRef.current); };
  }, [autoMode]);

  const runReplan = async (reason) => {
    setHealing(true);
    pushLog(`🔄 Self-healing replan: ${reason}`, "warn");
    try {
      const res = await fetch("/api/autonomous/replan", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          destination: form.destination, origin: form.origin,
          days: form.days, budget: form.budget, persona: form.persona,
          interests: form.services, reason,
          previousConfidence: confidence || 0,
        }),
      });
      const data = await res.json();
      if (data?.ok) {
        // Stream the agentic stages so user sees automation in action
        if (Array.isArray(data.stages)) {
          data.stages.forEach((s, i) => setTimeout(() => pushLog(s, "info"), i * 180));
        }
        if (data.itinerary) {
          // Actually swap the itinerary so the UI updates (map, plan, hotels...)
          setTimeout(() => {
            setItin(data.itinerary);
            setActiveDay(1);
          }, (data.stages?.length || 0) * 180);
        }
        const newConf = data.confidence || data.diff?.newConfidence || 0.9;
        const dlt = data.diff?.delta;
        setConf(newConf);
        pushLog(`✅ Replan complete · confidence ${Math.round(newConf*100)}%${dlt ? ` (${dlt>0?'+':''}${(dlt*100).toFixed(1)}%)` : ''}`, "success");
        addToast("Auto-replan complete — plan refreshed", "success");
      } else {
        pushLog(`Replan returned: ${data?.error || "no change"}`, "info");
      }
    } catch (e) { pushLog(`Replan failed: ${e.message}`, "error"); }
    finally { setHealing(false); }
  };

  // ── GPS / Nearby places ──────────────────────────────────────────────
  const [nearby, setNearby]         = useState(null);   // { items, origin }
  const [nearbyLoading, setNearbyLoading] = useState(false);
  const [nearbyCat, setNearbyCat]   = useState("all");

  const findNearbyAt = async (lat, lon, catOverride) => {
    const cat = catOverride || nearbyCat;
    setNearbyLoading(true);
    pushLog(`🛰️ Searching nearby @ ${lat.toFixed(4)}, ${lon.toFixed(4)} · category=${cat}`, "info");
    try {
      const r = await fetch(`/api/nearby?lat=${lat}&lon=${lon}&radius=3000&category=${cat}`);
      const d = await r.json();
      if (d.ok) {
        setNearby({ items: d.items || [], origin: d.origin, source: d.source, hint: d.hint });
        pushLog(`🗺️ Found ${d.count} nearby places (${d.source})`, "success");
        if (d.count > 0) addToast(`Found ${d.count} nearby places`, "success");
        else addToast("No nearby POIs found — try another category", "warning");
      } else {
        pushLog(`Nearby search failed: ${d.error || "unknown"}`, "error");
        addToast("No nearby places found", "warning");
      }
    } catch (e) {
      pushLog(`Nearby fetch error: ${e.message}`, "error");
      addToast(e.message, "error");
    } finally { setNearbyLoading(false); }
  };

  const findNearby = async (catOverride) => {
    const cat = catOverride || nearbyCat;
    if (!navigator.geolocation) {
      addToast("Geolocation not supported — using destination instead", "warning");
      if (itinerary?.destCoords) return findNearbyAt(itinerary.destCoords.lat, itinerary.destCoords.lon, cat);
      return;
    }
    setNearbyLoading(true);
    pushLog(`📍 Requesting GPS location · category=${cat}`, "info");
    addToast("Getting your location…", "info");
    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        const { latitude, longitude, accuracy } = pos.coords;
        pushLog(`📡 GPS lock · ${latitude.toFixed(4)}, ${longitude.toFixed(4)} · ±${Math.round(accuracy||0)}m`, "success");
        await findNearbyAt(latitude, longitude, cat);
      },
      (err) => {
        pushLog(`GPS denied: ${err.message}`, "error");
        if (itinerary?.destCoords) {
          addToast("GPS unavailable — searching near destination instead", "warning");
          findNearbyAt(itinerary.destCoords.lat, itinerary.destCoords.lon, cat);
        } else {
          addToast("Location permission denied — please allow GPS access or set a destination", "error");
          setNearbyLoading(false);
        }
      },
      { enableHighAccuracy: true, timeout: 12000, maximumAge: 60000 }
    );
  };

  const generate = async () => {
    if (!form.destination?.trim()) { addToast("Destination is required","error"); return; }
    if (!form.origin?.trim())      { addToast("Origin is required — where are you starting from?","error"); return; }
    setLoading(true);
    setAgentLog([]);
    setStage(0);
    setTaskQueue([
      { id:1, name:"Resolve geocode anchors",       status:"running" },
      { id:2, name:"Score POIs by proximity",       status:"queued" },
      { id:3, name:"Optimise daily budget split",   status:"queued" },
      { id:4, name:"MCTS route search · 200 iter",  status:"queued" },
      { id:5, name:"Fetch live transport inventory",status:"queued" },
      { id:6, name:"Rank hotels & cabs",            status:"queued" },
      { id:7, name:"Self-Critic verification",      status:"queued" },
    ]);
    pushLog(`🧭 Origin: ${form.origin} → Destination: ${form.destination}`, "info");
    pushLog("🤖 Spawning 13-agent autonomous pipeline…", "info");
    addToast("Building AI itinerary with real attractions...", "info");

    // Simulate live agent telemetry while the API request is in flight
    const stages = [
      "🔍 Scout Agent · resolving geocode anchors",
      "📍 Filtering POIs by proximity (≤120 km from destination)",
      "💸 Budget Optimiser · Double-Q with experience replay",
      "🗺️ Route Planner · MCTS UCB1-Tuned (200 iterations)",
      "📊 MDP Value Iteration · stateful daily flow",
      "🎲 Thompson Sampling · category preference",
      "☁️ Naive-Bayes weather classifier",
      "👥 GP crowd predictor · Gaussian Process surrogate",
      "✈️ Fetching real flights (6 airlines, 6 OTAs)",
      "🚆 Pulling IRCTC real train roster",
      "🏨 Curating real hotels + SRM-official options",
      "🚖 Local cab providers · 6 platforms",
      "🧠 Self-Critic + SHAP attribution",
    ];
    let stageIdx = 0;
    const stageTimer = setInterval(() => {
      if (stageIdx < stages.length) {
        pushLog(stages[stageIdx], "info");
        setStage(Math.min(PIPELINE.length - 1, Math.floor((stageIdx / stages.length) * PIPELINE.length)));
        // Roll the task queue forward
        setTaskQueue(q => q.map((t, i) => {
          const target = Math.floor((stageIdx / stages.length) * q.length);
          if (i < target) return { ...t, status: "done" };
          if (i === target) return { ...t, status: "running" };
          return t;
        }));
        stageIdx++;
      }
    }, 220);

    try {
      const res = await fetch("/api/itinerary", {
        method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify({
          destination:    form.destination,
          origin:         form.origin,
          number_of_days: form.days,
          budget:         form.budget,
          interests:      form.services,
          persona:        form.persona,
        }),
      });
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || "Failed");
      setItin(data.itinerary); setActiveDay(1);

      // Estimate confidence from coverage signals
      const has = (k) => Array.isArray(data.itinerary[k]) && data.itinerary[k].length > 0;
      const score = 0.55
        + (has("days") ? 0.1 : 0)
        + (has("flights") ? 0.07 : 0)
        + (has("trains") ? 0.07 : 0)
        + (has("hotels") ? 0.07 : 0)
        + (has("cabs") ? 0.04 : 0)
        + (has("restaurants") ? 0.04 : 0)
        + (has("weather") ? 0.04 : 0)
        + (has("safetyTips") ? 0.03 : 0);
      const conf = Math.min(0.97, score);
      setConf(conf);

      pushLog(`✅ Pipeline complete · ${data.itinerary.totalAttractions} stops · confidence ${Math.round(conf*100)}%`, "success");
      pushLog(`✈ ${data.itinerary.flights?.length||0} flights · 🚆 ${data.itinerary.trains?.length||0} trains · 🏨 ${data.itinerary.hotels?.length||0} hotels · 🚖 ${data.itinerary.cabs?.length||0} cab options`, "success");
      addToast(`${form.days}-day itinerary ready · ${data.itinerary.totalAttractions} stops mapped!`, "success");
      // Mark all queue tasks as done
      setStage(PIPELINE.length - 1);
      setTaskQueue(q => q.map(t => ({ ...t, status:"done" })));

      // Self-healing: if confidence low and autoMode enabled, trigger replan
      if (autoMode && conf < 0.78) {
        runReplan(`Low confidence ${Math.round(conf*100)}% — auto-improving`);
      }
    } catch(e) {
      pushLog(`❌ ${e.message}`, "error");
      addToast(e.message, "error");
      if (autoMode) runReplan(`Initial generation failed: ${e.message}`);
    } finally {
      clearInterval(stageTimer);
      setLoading(false);
    }
  };

  const TABS = [
    { id:"plan",      label:"Day Plan" },
    { id:"map",       label:"Map View" },
    { id:"nearby",    label:"📍 Nearby" },
    { id:"flights",   label:"Flights" },
    { id:"trains",    label:"Trains" },
    { id:"hotels",    label:"Hotels" },
    { id:"cabs",      label:"Cabs" },
    { id:"food",      label:"Food" },
    { id:"language",  label:"Language" },
    { id:"weather",   label:"Weather" },
    { id:"safety",    label:"Safety" },
    { id:"packing",   label:"Packing" },
    { id:"crowd",     label:"Crowd" },
    { id:"checklist", label:"Checklist" },
  ];

  return (
    <div>
      <div className="section-title">Itinerary Planner</div>
      <div className="section-sub">AI-powered day-by-day plans · real attractions · live map · weather · language · packing · safety</div>

      <div className="itinerary-page-grid">
        <div style={{ display:"flex", flexDirection:"column", gap:18 }}>
          {/* Config */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
                Plan Configuration
              </span>
              <span className="pill blue">AI-Powered</span>
            </div>
            <div className="card-body" style={{ display:"flex", flexDirection:"column", gap:16 }}>
              <div>
                <div className="field-label" style={{ marginBottom:8 }}>Travel Persona</div>
                <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:8 }}>
                  {PERSONAS.map(p=>(
                    <button key={p.id} onClick={()=>setForm(f=>({...f,persona:p.id}))}
                      style={{ padding:"10px 8px", borderRadius:"var(--r-md)", border:`1.5px solid ${form.persona===p.id?"var(--blue)":"var(--border)"}`,
                        background: form.persona===p.id?"var(--blue-dim)":"var(--bg-soft)", cursor:"pointer",
                        color: form.persona===p.id?"var(--blue)":"var(--text-2)", transition:"all 0.15s",
                        display:"flex", flexDirection:"column", alignItems:"center", gap:4,
                      }}>
                      <span style={{ fontSize:13, fontWeight:700 }}>{p.label}</span>
                      <span style={{ fontSize:10.5, opacity:0.7 }}>{p.sub}</span>
                    </button>
                  ))}
                </div>
              </div>
              <div className="grid-2">
                <div className="field-group">
                  <label className="field-label">Origin (where from?)</label>
                  <input className="field-input" value={form.origin} placeholder="e.g. SRMIST Kattankulathur"
                    onChange={e=>setForm(f=>({...f,origin:e.target.value}))} />
                </div>
                <div className="field-group">
                  <label className="field-label">Destination</label>
                  <input className="field-input" value={form.destination} placeholder="e.g. Goa"
                    onChange={e=>setForm(f=>({...f,destination:e.target.value}))} />
                </div>
                <div className="field-group">
                  <label className="field-label">Days</label>
                  <input type="number" className="field-input" min={1} max={10} value={form.days}
                    onChange={e=>setForm(f=>({...f,days:+e.target.value}))} />
                </div>
                <div className="field-group">
                  <label className="field-label">Budget (₹)</label>
                  <input type="number" className="field-input" step={500} value={form.budget}
                    onChange={e=>setForm(f=>({...f,budget:+e.target.value}))} />
                </div>
              </div>
              <div>
                <div className="field-label" style={{ marginBottom:8 }}>Services</div>
                <div style={{ display:"flex", flexWrap:"wrap", gap:6 }}>
                  {SERVICES.map(s=>(
                    <button key={s} className={form.services?.includes(s)?"chip active":"chip"}
                      onClick={()=>setForm(f=>({ ...f, services: f.services?.includes(s) ? f.services.filter(x=>x!==s) : [...(f.services||[]),s] }))}>
                      {s}
                    </button>
                  ))}
                </div>
              </div>
              <button className="btn btn-primary w-full" onClick={generate} disabled={loading}>
                {loading ? "Generating..." : "Generate Itinerary →"}
              </button>
            </div>
          </div>

          {/* Feature tabs */}
          <div className="card">
            <div style={{ display:"flex", gap:0, borderBottom:"1px solid var(--border)", overflowX:"auto" }}>
              {TABS.map(t=>(
                <button key={t.id} onClick={()=>setActiveTab(t.id)}
                  style={{ padding:"12px 16px", border:"none", cursor:"pointer", fontSize:13, fontWeight:activeTab===t.id?600:500,
                    color:activeTab===t.id?"var(--blue)":"var(--text-2)", background:"transparent", whiteSpace:"nowrap",
                    borderBottom:`2px solid ${activeTab===t.id?"var(--blue)":"transparent"}`, marginBottom:-1, transition:"all 0.15s" }}>
                  {t.label}
                </button>
              ))}
            </div>

            <div style={{ padding:0 }}>
              {activeTab==="plan" && (
                <div style={{ padding:"16px 20px" }}>
                  {!itinerary ? (
                    <div style={{ textAlign:"center", padding:"24px", color:"var(--text-2)", fontSize:13 }}>
                      Generate an itinerary above to see your day-by-day plan here.
                    </div>
                  ) : (
                    <>
                      {itinerary.heroImage && (
                        <motion.div initial={{opacity:0, y:-8}} animate={{opacity:1, y:0}}
                          style={{ position:"relative", height:180, borderRadius:14, overflow:"hidden", marginBottom:14, boxShadow:"var(--shadow)" }}>
                          <img src={itinerary.heroImage} alt={itinerary.destination}
                            onError={(e)=>{ e.currentTarget.src = flickrUrl(itinerary.destination + " travel landmark", 1600, 900); }}
                            style={{ width:"100%", height:"100%", objectFit:"cover", display:"block" }} />
                          <div style={{ position:"absolute", inset:0, background:"linear-gradient(180deg, rgba(0,0,0,0) 40%, rgba(0,0,0,0.75) 100%)" }} />
                          <div style={{ position:"absolute", left:18, bottom:14, right:18, color:"#fff" }}>
                            <div style={{ fontSize:22, fontWeight:700, textShadow:"0 2px 12px rgba(0,0,0,0.6)" }}>
                              {itinerary.destination}
                            </div>
                            <div style={{ fontSize:12, opacity:0.92, marginTop:2, textShadow:"0 1px 6px rgba(0,0,0,0.8)" }}>
                              {itinerary.days.length} days · {itinerary.totalAttractions || itinerary.allStops?.length || 0} curated stops
                              {itinerary.route?.distanceKm && ` · ${itinerary.route.distanceKm} km from ${itinerary.origin}`}
                            </div>
                          </div>
                        </motion.div>
                      )}
                      <p style={{ fontSize:13, color:"var(--text-2)", marginBottom:14 }}>{itinerary.summary}</p>
                      <div style={{ display:"flex", gap:6, flexWrap:"wrap", marginBottom:16 }}>
                        {itinerary.days.map(d=>(
                          <button key={d.day} className={activeDay===d.day?"chip active":"chip"} onClick={()=>setActiveDay(d.day)}>
                            Day {d.day} {d.weather && <span style={{ marginLeft:4 }}>{d.weather.emoji}</span>}
                          </button>
                        ))}
                      </div>
                      <div className="timeline">
                        {itinerary.days.filter(d=>d.day===activeDay).map(day=>(
                          <div key={day.day}>
                            <div className="timeline-day-header">
                              <div className="timeline-day-num">{day.day}</div>
                              <div style={{ flex:1 }}>
                                <div className="timeline-day-title">{day.theme}</div>
                                {day.weather && (
                                  <div style={{ fontSize:11.5, color:"var(--text-3)", marginTop:2 }}>
                                    {day.weather.emoji} {day.weather.max}°/{day.weather.min}°
                                    {day.weather.precipitation>0 && ` · ${day.weather.precipitation}% rain`}
                                    · ₹{day.dailyBudget?.toLocaleString("en-IN")} day budget
                                  </div>
                                )}
                              </div>
                            </div>
                            <div className="timeline-stops">
                              {day.plan.map((stop,i)=>{
                                const meta = STOP_COLORS[stop.type] || STOP_COLORS.Activity;
                                const photo = stop.thumbnail || stop.image;
                                return (
                                  <motion.div key={i} className={`timeline-stop ${i===0?"highlight":""}`}
                                    initial={{opacity:0,x:-8}} animate={{opacity:1,x:0}} transition={{delay:i*0.06}}
                                    whileHover={{ scale: 1.01, x: 2 }}>
                                    <div className="stop-time">{stop.time || `${8+i*2}:00`}</div>
                                    <div className="stop-body" style={{ display:"flex", gap:12, alignItems:"flex-start" }}>
                                      {photo && (
                                        <a href={stop.image || photo} target="_blank" rel="noopener noreferrer"
                                          style={{ flexShrink:0, display:"block", width:104, height:78, borderRadius:10, overflow:"hidden", background:"var(--bg-2)", border:"1px solid var(--border)", boxShadow:"var(--shadow-sm)" }}>
                                          <img src={photo} alt={stop.name} loading="lazy"
                                            onError={(e)=>{ e.currentTarget.src = flickrUrl(stop.name + " " + (itinerary?.destination||""), 600, 400); }}
                                            style={{ width:"100%", height:"100%", objectFit:"cover", display:"block", transition:"transform 0.3s" }}
                                            onMouseOver={(e)=>{ e.currentTarget.style.transform="scale(1.08)"; }}
                                            onMouseOut={(e)=>{ e.currentTarget.style.transform="scale(1)"; }} />
                                        </a>
                                      )}
                                      <div style={{ flex:1, minWidth:0 }}>
                                        <div style={{ display:"flex", alignItems:"center", gap:6, marginBottom:2, flexWrap:"wrap" }}>
                                          <span style={{ fontSize:11, background:meta.bg, color:meta.color, padding:"2px 7px", borderRadius:"var(--r-full)", fontWeight:600 }}>{stop.type}</span>
                                          <div className="stop-title">{stop.name}</div>
                                        </div>
                                        {stop.extract && (
                                          <div style={{ fontSize:12, color:"var(--text-2)", lineHeight:1.45, marginTop:4, marginBottom:4 }}>
                                            {stop.extract.length > 180 ? stop.extract.slice(0,180)+"…" : stop.extract}
                                          </div>
                                        )}
                                        <div className="stop-detail">{stop.note}</div>
                                        <div style={{ display:"flex", gap:8, marginTop:6, flexWrap:"wrap" }}>
                                          {stop.mapsUrl && (
                                            <a href={stop.mapsUrl} target="_blank" rel="noopener noreferrer"
                                              style={{ fontSize:11, color:"var(--blue)", fontWeight:600, textDecoration:"none" }}>
                                              📍 Maps
                                            </a>
                                          )}
                                          {(stop.wikiUrl || stop.wikiTitle) && (
                                            <a href={stop.wikiUrl || `https://en.wikipedia.org/wiki/${stop.wikiTitle}`} target="_blank" rel="noopener noreferrer"
                                              style={{ fontSize:11, color:"var(--purple)", fontWeight:600, textDecoration:"none" }}>
                                              📖 Wikipedia
                                            </a>
                                          )}
                                          {stop.bookingUrl && (
                                            <a href={stop.bookingUrl} target="_blank" rel="noopener noreferrer"
                                              style={{ fontSize:11, color:"var(--green)", fontWeight:600, textDecoration:"none" }}>
                                              🔗 Book / Reserve
                                            </a>
                                          )}
                                          {stop.cost && (
                                            <span style={{ fontSize:11, color:"var(--amber)", fontWeight:600 }}>
                                              ₹{Number(stop.cost).toLocaleString("en-IN")}
                                            </span>
                                          )}
                                        </div>
                                      </div>
                                    </div>
                                  </motion.div>
                                );
                              })}
                            </div>
                          </div>
                        ))}
                      </div>
                    </>
                  )}
                </div>
              )}

              {activeTab==="map" && (
                <div style={{ padding:"16px 20px" }}>
                  {!itinerary ? (
                    <div style={{ textAlign:"center", padding:"24px", color:"var(--text-2)", fontSize:13 }}>
                      Generate an itinerary to see all places on the map.
                    </div>
                  ) : (
                    <>
                      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:12, flexWrap:"wrap", gap:8 }}>
                        <div style={{ fontSize:13, color:"var(--text-2)" }}>
                          {itinerary.allStops?.length || 0} stops · {itinerary.days.length} days
                          {itinerary.originCoords && ` · from ${itinerary.originCoords.name?.split(",")[0]}`}
                        </div>
                        <div style={{ display:"flex", gap:6, flexWrap:"wrap" }}>
                          <button className={mapAllDays?"chip active":"chip"} onClick={()=>setMapAll(true)}>All days</button>
                          {itinerary.days.map(d=>(
                            <button key={d.day} className={!mapAllDays && activeDay===d.day?"chip active":"chip"}
                              onClick={()=>{ setMapAll(false); setActiveDay(d.day); }}>
                              D{d.day}
                            </button>
                          ))}
                        </div>
                      </div>
                      <ItineraryMap itinerary={itinerary} height={460} activeDay={mapAllDays ? null : activeDay} />
                      <div style={{ marginTop:10, fontSize:11.5, color:"var(--text-3)", display:"flex", gap:14, flexWrap:"wrap" }}>
                        <span>🔵 Origin</span><span>🔴 Destination centre</span>
                        <span>📍 Day-coloured stops · click for details</span>
                      </div>
                    </>
                  )}
                </div>
              )}

              {activeTab==="nearby" && (
                <div style={{ padding:"16px 20px" }}>
                  <div style={{ display:"flex", flexWrap:"wrap", gap:8, alignItems:"center", marginBottom:12 }}>
                    <button className="btn btn-primary btn-sm" onClick={()=>findNearby()} disabled={nearbyLoading}>
                      {nearbyLoading ? "Locating…" : "📍 Use my GPS"}
                    </button>
                    {itinerary?.destCoords && (
                      <button className="btn btn-secondary btn-sm" onClick={()=>findNearbyAt(itinerary.destCoords.lat, itinerary.destCoords.lon)} disabled={nearbyLoading}>
                        🏝️ Near destination
                      </button>
                    )}
                    <div style={{ display:"flex", gap:6, flexWrap:"wrap" }}>
                      {[
                        ["all","All"], ["attractions","Attractions"],
                        ["food","Food"], ["hospital","Medical"], ["parks","Parks"],
                        ["shops","Shops"], ["fuel","Fuel/EV"],
                      ].map(([k,l])=>(
                        <button key={k}
                          className={nearbyCat===k?"chip active":"chip"}
                          onClick={()=>{ setNearbyCat(k); if(nearby) findNearbyAt(nearby.origin.lat, nearby.origin.lon, k); }}>
                          {l}
                        </button>
                      ))}
                    </div>
                  </div>
                  {!nearby ? (
                    <div style={{ color:"var(--text-3)", fontSize:13, padding:"20px 0", textAlign:"center" }}>
                      Tap <b>📍 Use my GPS</b> for real attractions, restaurants and services within 3 km.
                      <br/><span style={{ fontSize:11.5 }}>Or hit <b>🏝️ Near destination</b> to explore around your itinerary's destination.</span>
                    </div>
                  ) : nearby.items.length === 0 ? (
                    <div style={{ color:"var(--text-3)", fontSize:13, padding:"16px 0" }}>
                      No places found within 3 km — try a different category, widen the radius, or retry (POI providers may be temporarily slow).
                      {nearby.hint && <div style={{ marginTop:6, fontSize:11.5, color:"var(--amber)" }}>⚠ {nearby.hint}</div>}
                    </div>
                  ) : (
                    <div>
                      <div style={{ fontSize:12, color:"var(--text-3)", marginBottom:10 }}>
                        {nearby.items.length} places · @ {nearby.origin?.lat?.toFixed(4)}, {nearby.origin?.lon?.toFixed(4)} · source: <span style={{ color:"var(--green)", fontWeight:600 }}>{nearby.source}</span>
                      </div>
                      <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fill,minmax(260px,1fr))", gap:12 }}>
                        {nearby.items.map((p,i)=>(
                          <motion.div key={p.id||i}
                            initial={{ opacity:0, y:8 }} animate={{ opacity:1, y:0 }} transition={{ delay:i*0.03 }}
                            whileHover={{ y:-3, boxShadow:"var(--shadow-lg)" }}
                            style={{ border:"1.5px solid var(--border)", borderRadius:"var(--r-md)", background:"var(--bg-soft)", overflow:"hidden", cursor:"pointer", transition:"all 0.2s" }}>
                            {p.thumbnail && (
                              <div style={{ width:"100%", height:130, overflow:"hidden", background:"var(--bg-2)", position:"relative" }}>
                                <img src={p.thumbnail} alt={p.name} loading="lazy"
                                  onError={(e)=>{ e.currentTarget.src = flickrUrl(p.name + " " + (p.type||"place"), 600, 400); }}
                                  style={{ width:"100%", height:"100%", objectFit:"cover", display:"block", transition:"transform 0.4s" }}
                                  onMouseOver={(e)=>{ e.currentTarget.style.transform="scale(1.06)"; }}
                                  onMouseOut={(e)=>{ e.currentTarget.style.transform="scale(1)"; }} />
                                <div style={{ position:"absolute", top:6, right:6, padding:"3px 8px", borderRadius:"var(--r-full)", background:"rgba(0,0,0,0.65)", color:"#fff", fontSize:11, fontWeight:600, backdropFilter:"blur(6px)" }}>
                                  ⭐ {Number(p.rating).toFixed(1)}
                                </div>
                                <div style={{ position:"absolute", bottom:6, left:6, padding:"2px 7px", borderRadius:"var(--r-full)", background:"rgba(0,0,0,0.65)", color:"#fff", fontSize:10, fontWeight:600 }}>
                                  {p.distance < 1000 ? `${p.distance} m` : `${(p.distance/1000).toFixed(1)} km`}
                                </div>
                              </div>
                            )}
                            <div style={{ padding:"10px 12px" }}>
                              <div style={{ fontSize:13.5, fontWeight:700, color:"var(--text)" }}>{p.name}</div>
                              <div style={{ fontSize:10.5, color:"var(--text-3)", textTransform:"uppercase", letterSpacing:"0.04em", marginTop:3 }}>{p.type}</div>
                              {p.description && <div style={{ fontSize:11.5, color:"var(--text-2)", marginTop:4, lineHeight:1.4 }}>{p.description}</div>}
                              {p.opening_hours && <div style={{ fontSize:11, color:"var(--text-3)", marginTop:3 }}>🕐 {p.opening_hours}</div>}
                              {p.address && <div style={{ fontSize:11, color:"var(--text-3)", marginTop:3 }}>📌 {p.address}</div>}
                              <div style={{ display:"flex", gap:6, marginTop:8, flexWrap:"wrap" }}>
                                <a href={p.directionsUrl} target="_blank" rel="noopener noreferrer"
                                   style={{ fontSize:11, padding:"3px 8px", background:"var(--blue-dim)", color:"var(--blue)", borderRadius:"var(--r-full)", fontWeight:600, textDecoration:"none" }}>
                                  🧭 Directions
                                </a>
                                <a href={p.mapsUrl} target="_blank" rel="noopener noreferrer"
                                   style={{ fontSize:11, padding:"3px 8px", background:"var(--purple-bg)", color:"var(--purple)", borderRadius:"var(--r-full)", fontWeight:600, textDecoration:"none" }}>
                                  Maps →
                                </a>
                                {p.website && (
                                  <a href={p.website} target="_blank" rel="noopener noreferrer"
                                     style={{ fontSize:11, padding:"3px 8px", background:"var(--green-bg)", color:"var(--green)", borderRadius:"var(--r-full)", fontWeight:600, textDecoration:"none" }}>
                                    🌐 Website
                                  </a>
                                )}
                                {p.phone && (
                                  <a href={`tel:${p.phone}`}
                                     style={{ fontSize:11, padding:"3px 8px", background:"var(--amber-bg)", color:"var(--amber)", borderRadius:"var(--r-full)", fontWeight:600, textDecoration:"none" }}>
                                    📞 Call
                                  </a>
                                )}
                              </div>
                            </div>
                          </motion.div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {activeTab==="flights" && (
                <div style={{ padding:"16px 20px" }}>
                  {!itinerary?.flights?.length ? (
                    <div style={{ color:"var(--text-3)", fontSize:13 }}>
                      {!itinerary ? "Generate the itinerary first." : "No flights for this route — try a different origin or check trains/cabs."}
                    </div>
                  ) : (
                    <div>
                      <div style={{ fontSize:13, color:"var(--text-2)", marginBottom:10 }}>
                        {itinerary.route ? <>✈ {itinerary.route.fromIATA || itinerary.route.from} → {itinerary.route.toIATA || itinerary.route.to} · ~{itinerary.route.distanceKm} km · sorted cheapest first</> : "Real airline schedules with one-click booking on multiple platforms"}
                      </div>
                      <div style={{ display:"flex", flexDirection:"column", gap:10 }}>
                        {itinerary.flights.map((f,i)=>(
                          <div key={i} style={{ padding:"12px 14px", border:"1.5px solid var(--border)", borderRadius:"var(--r-md)", background:"var(--bg-soft)" }}>
                            <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start", flexWrap:"wrap", gap:8 }}>
                              <div>
                                <div style={{ fontSize:14, fontWeight:700, color:"var(--text)" }}>{f.airline} <span style={{ color:"var(--text-3)", fontWeight:500, fontSize:12, marginLeft:6 }}>{f.flight_no}</span></div>
                                <div style={{ fontSize:12, color:"var(--text-2)", marginTop:3 }}>
                                  {f.origin_code || ""} {f.departure} → {f.dest_code || ""} {f.arrival} · {f.duration} · {f.stops===0?"non-stop":`${f.stops} stop`} · {f.class}
                                </div>
                                <div style={{ fontSize:11, color:"var(--text-3)", marginTop:2 }}>{f.aircraft} · ⭐ {f.rating}</div>
                              </div>
                              <div style={{ textAlign:"right" }}>
                                <div style={{ fontFamily:"'Sora',sans-serif", fontWeight:800, fontSize:18, color:"var(--text)" }}>₹{f.price?.toLocaleString("en-IN")}</div>
                              </div>
                            </div>
                            <div style={{ display:"flex", flexWrap:"wrap", gap:6, marginTop:10 }}>
                              {f.bookingPlatforms?.map((p,j)=>(
                                <a key={j} href={p.url} target="_blank" rel="noopener noreferrer"
                                  style={{ fontSize:11.5, padding:"4px 10px", background:"var(--blue-dim)", color:"var(--blue)", borderRadius:"var(--r-full)", fontWeight:600, textDecoration:"none" }}>
                                  {p.name} →
                                </a>
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {activeTab==="trains" && (
                <div style={{ padding:"16px 20px" }}>
                  {!itinerary?.trains?.length ? (
                    <div style={{ color:"var(--text-3)", fontSize:13 }}>
                      {!itinerary ? "Generate the itinerary first." : "No direct trains found — check flights or cabs above."}
                    </div>
                  ) : (
                    <div>
                      <div style={{ fontSize:13, color:"var(--text-2)", marginBottom:10 }}>🚆 IRCTC roster · sorted cheapest first · multi-platform booking</div>
                      <div style={{ display:"flex", flexDirection:"column", gap:10 }}>
                        {itinerary.trains.map((t,i)=>(
                          <div key={i} style={{ padding:"12px 14px", border:"1.5px solid var(--border)", borderRadius:"var(--r-md)", background:"var(--bg-soft)" }}>
                            <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start", flexWrap:"wrap", gap:8 }}>
                              <div>
                                <div style={{ fontSize:14, fontWeight:700, color:"var(--text)" }}>{t.train_name} <span style={{ color:"var(--text-3)", fontWeight:500, fontSize:12, marginLeft:6 }}>#{t.train_no}</span></div>
                                <div style={{ fontSize:12, color:"var(--text-2)", marginTop:3 }}>
                                  Departs {t.departure} · {t.duration} · {t.class} ({t.available_classes?.join(", ")})
                                </div>
                              </div>
                              <div style={{ textAlign:"right" }}>
                                <div style={{ fontFamily:"'Sora',sans-serif", fontWeight:800, fontSize:18, color:"var(--text)" }}>₹{t.price?.toLocaleString("en-IN")}</div>
                              </div>
                            </div>
                            <div style={{ display:"flex", flexWrap:"wrap", gap:6, marginTop:10 }}>
                              {t.bookingPlatforms?.map((p,j)=>(
                                <a key={j} href={p.url} target="_blank" rel="noopener noreferrer"
                                  style={{ fontSize:11.5, padding:"4px 10px", background:"var(--green-bg)", color:"var(--green)", borderRadius:"var(--r-full)", fontWeight:600, textDecoration:"none" }}>
                                  {p.name} →
                                </a>
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {activeTab==="hotels" && (
                <div style={{ padding:"16px 20px" }}>
                  {!itinerary?.hotels?.length ? (
                    <div style={{ color:"var(--text-3)", fontSize:13 }}>Generate the itinerary first.</div>
                  ) : (
                    <div>
                      <div style={{ fontSize:13, color:"var(--text-2)", marginBottom:10 }}>🏨 Real curated hotels in {itinerary.destination} · {itinerary.isSrmDestination ? "SRM-official options pinned at top" : "sorted by price"}</div>
                      <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fill,minmax(280px,1fr))", gap:10 }}>
                        {itinerary.hotels.map((h,i)=>(
                          <div key={i} style={{ padding:"12px 14px", border:`1.5px solid ${h.srmOfficial?"#f59e0b":"var(--border)"}`, borderRadius:"var(--r-md)", background: h.srmOfficial?"linear-gradient(135deg,#fffbeb,#fef3c7)":"var(--bg-soft)" }}>
                            <div style={{ fontSize:14, fontWeight:700, color:"var(--text)" }}>
                              {h.srmOfficial && <span style={{ marginRight:5 }}>⭐</span>}{h.name}
                            </div>
                            <div style={{ fontSize:11.5, color:"var(--text-3)", marginTop:3 }}>{h.address} · {Array(h.stars||0).fill("★").join("")} · ⭐ {h.rating}</div>
                            {h.description && <div style={{ fontSize:11.5, color:"var(--text-2)", marginTop:4, lineHeight:1.4 }}>{h.description}</div>}
                            <div style={{ marginTop:8, display:"flex", justifyContent:"space-between", alignItems:"center", flexWrap:"wrap", gap:6 }}>
                              <div>
                                {h.applyRequired ? (
                                  <span style={{ fontSize:12, color:"#b45309", fontWeight:700 }}>Apply via SRMIST portal</span>
                                ) : (
                                  <span style={{ fontFamily:"'Sora',sans-serif", fontWeight:800, fontSize:16 }}>₹{h.price_per_night?.toLocaleString("en-IN")}<span style={{ fontSize:11, color:"var(--text-3)", fontWeight:500 }}>/night</span></span>
                                )}
                              </div>
                            </div>
                            <div style={{ display:"flex", flexWrap:"wrap", gap:5, marginTop:8 }}>
                              {h.bookingPlatforms?.map((p,j)=>(
                                <a key={j} href={p.url} target="_blank" rel="noopener noreferrer"
                                  style={{ fontSize:11, padding:"3px 8px", background: p.srmOfficial?"#f59e0b":"var(--purple-bg)", color: p.srmOfficial?"#fff":"var(--purple)", borderRadius:"var(--r-full)", fontWeight:600, textDecoration:"none" }}>
                                  {p.name} →
                                </a>
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {activeTab==="cabs" && (
                <div style={{ padding:"16px 20px" }}>
                  {!itinerary?.cabs?.length ? (
                    <div style={{ color:"var(--text-3)", fontSize:13 }}>Generate the itinerary first.</div>
                  ) : (
                    <div>
                      <div style={{ fontSize:13, color:"var(--text-2)", marginBottom:10 }}>🚖 Local cab providers · estimated 10-km / 20-km fares</div>
                      <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fill,minmax(220px,1fr))", gap:10 }}>
                        {itinerary.cabs.map((c,i)=>(
                          <div key={i} style={{ padding:"12px 14px", border:"1.5px solid var(--border)", borderRadius:"var(--r-md)", background:"var(--bg-soft)" }}>
                            <div style={{ fontSize:14, fontWeight:700, color:"var(--text)" }}>{c.provider} <span style={{ fontSize:11, color:"var(--text-3)", fontWeight:500 }}>· {c.type}</span></div>
                            <div style={{ fontSize:11, color:"var(--text-3)", marginTop:3 }}>{c.provider_about}</div>
                            <div style={{ fontSize:12, color:"var(--text-2)", marginTop:6 }}>
                              ₹{c.base_fare} base · ₹{c.price_per_km}/km · ⭐ {c.rating}
                            </div>
                            <div style={{ fontSize:12, color:"var(--text-2)", marginTop:3 }}>~₹{c.estimated_10km} for 10 km · ~₹{c.estimated_20km} for 20 km</div>
                            <div style={{ display:"flex", flexWrap:"wrap", gap:6, marginTop:8 }}>
                              {c.bookingPlatforms?.map((p,j)=>(
                                <a key={j} href={p.url} target="_blank" rel="noopener noreferrer"
                                  style={{ fontSize:11, padding:"3px 8px", background:"var(--amber-bg)", color:"var(--amber)", borderRadius:"var(--r-full)", fontWeight:600, textDecoration:"none" }}>
                                  {p.name} →
                                </a>
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {activeTab==="food" && (
                <div style={{ padding:"16px 20px" }}>
                  {!itinerary ? <div style={{ color:"var(--text-3)", fontSize:13 }}>Generate an itinerary first to see food recommendations.</div> : (
                    <div>
                      <div style={{ fontSize:14, fontWeight:700, color:"var(--text)", marginBottom:10 }}>
                        Curated restaurants in {itinerary.destination}
                      </div>
                      <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fill,minmax(240px,1fr))", gap:10 }}>
                        {itinerary.restaurants?.map((r,i)=>(
                          <a key={i} href={r.bookingUrl} target="_blank" rel="noopener noreferrer"
                            style={{ display:"block", padding:"12px 14px", border:"1.5px solid var(--border)", borderRadius:"var(--r-md)", textDecoration:"none", color:"inherit", background:"var(--bg-soft)" }}>
                            <div style={{ fontWeight:700, fontSize:14, color:"var(--text)" }}>{r.name}</div>
                            <div style={{ fontSize:11.5, color:"var(--text-3)", marginTop:3 }}>{r.cuisine} · {r.price_range} · ⭐ {r.rating}</div>
                            <div style={{ fontSize:11.5, color:"var(--green)", fontWeight:600, marginTop:4 }}>~₹{r.avgCost} pp · Zomato →</div>
                          </a>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {activeTab==="weather" && (
                <div style={{ padding:"16px 20px" }}>
                  {!itinerary?.weather?.length ? <div style={{ color:"var(--text-3)", fontSize:13 }}>No weather data — generate the itinerary first.</div> : (
                    <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fit,minmax(110px,1fr))", gap:10 }}>
                      {itinerary.weather.map((w,i)=>(
                        <div key={i} style={{ padding:"12px", border:"1.5px solid var(--border)", borderRadius:"var(--r-md)", textAlign:"center", background:"var(--bg-soft)" }}>
                          <div style={{ fontSize:24 }}>{w.emoji}</div>
                          <div style={{ fontFamily:"'Sora',sans-serif", fontWeight:800, fontSize:17 }}>{w.max}°</div>
                          <div style={{ fontSize:11, color:"var(--text-3)" }}>{w.min}° · {w.label}</div>
                          {w.precipitation > 0 && <div style={{ fontSize:10.5, color:"var(--blue)" }}>{w.precipitation}% rain</div>}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {activeTab==="safety" && (
                <div style={{ padding:"16px 20px" }}>
                  {!itinerary ? <div style={{ color:"var(--text-3)", fontSize:13 }}>Generate the itinerary first.</div> : (
                    <div>
                      <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fit,minmax(150px,1fr))", gap:8, marginBottom:14 }}>
                        {Object.entries(itinerary.emergency || {}).map(([k,v])=>(
                          <div key={k} style={{ padding:"8px 10px", background:"var(--red-bg)", borderRadius:"var(--r-sm)", border:"1.5px solid rgba(220,38,38,0.2)" }}>
                            <div style={{ fontSize:10, fontWeight:700, color:"var(--red)", textTransform:"uppercase", letterSpacing:"0.05em" }}>{k.replace(/_/g," ")}</div>
                            <div style={{ fontSize:13, fontWeight:700, color:"var(--text)", marginTop:2 }}>{v}</div>
                          </div>
                        ))}
                      </div>
                      <ul style={{ listStyle:"none", padding:0, margin:0, display:"flex", flexDirection:"column", gap:8 }}>
                        {itinerary.safetyTips?.map((t,i)=>(
                          <li key={i} style={{ display:"flex", gap:8, fontSize:13, color:"var(--text-2)" }}>
                            <span style={{ color:"var(--green)", flexShrink:0 }}>✓</span>{t}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}

              {activeTab==="packing" && (
                <div style={{ padding:"16px 20px" }}>
                  {!itinerary?.packingList ? <div style={{ color:"var(--text-3)", fontSize:13 }}>Generate the itinerary first.</div> : (
                    <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fit,minmax(220px,1fr))", gap:14 }}>
                      {Object.entries(itinerary.packingList).map(([cat, items])=>(
                        <div key={cat} style={{ padding:"12px 14px", border:"1.5px solid var(--border)", borderRadius:"var(--r-md)", background:"var(--bg-soft)" }}>
                          <div style={{ fontSize:13, fontWeight:700, color:"var(--text)", marginBottom:8 }}>{cat}</div>
                          <ul style={{ listStyle:"none", padding:0, margin:0, display:"flex", flexDirection:"column", gap:5 }}>
                            {items.map((it,i)=>(
                              <li key={i} style={{ fontSize:12, color:"var(--text-2)", display:"flex", gap:6 }}>
                                <span style={{ color:"var(--blue)" }}>□</span>{it}
                              </li>
                            ))}
                          </ul>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {activeTab==="crowd" && <div style={{ padding:"16px 20px" }}><CrowdAnalyzer destination={form.destination} addToast={addToast} /></div>}
              {activeTab==="language" && (
                <div style={{ padding:"16px 20px" }}>
                  {itinerary?.languageTips ? (
                    <div>
                      <div style={{ fontSize:14, fontWeight:700, color:"var(--text)", marginBottom:10 }}>
                        {itinerary.languageTips.language} essentials
                      </div>
                      <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fill,minmax(220px,1fr))", gap:8 }}>
                        {itinerary.languageTips.phrases?.map((p,i)=>(
                          <div key={i} style={{ padding:"10px 12px", border:"1.5px solid var(--border)", borderRadius:"var(--r-md)", background:"var(--bg-soft)" }}>
                            <div style={{ fontWeight:700, fontSize:14, color:"var(--text)" }}>{p.phrase}</div>
                            <div style={{ fontSize:12, color:"var(--text-2)" }}>{p.meaning}</div>
                            <div style={{ fontSize:11, color:"var(--text-3)", fontFamily:"monospace", marginTop:2 }}>/{p.pronunciation}/</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : <LanguageTips destination={form.destination} />}
                </div>
              )}
              {activeTab==="checklist" && <div style={{ padding:"16px 20px" }}><TripChecklist destination={form.destination} persona={form.persona} days={form.days} addToast={addToast} /></div>}
            </div>
          </div>
        </div>

        {/* Right sidebar */}
        <div style={{ display:"flex", flexDirection:"column", gap:14 }}>
          {/* Agentic AI Automation Panel */}
          <div className="card" style={{ border:"1.5px solid var(--blue)", background:"linear-gradient(135deg, rgba(37,99,235,0.04), rgba(124,58,237,0.04))" }}>
            <div className="card-header">
              <span className="card-title" style={{ display:"flex", alignItems:"center", gap:6 }}>
                <span style={{ display:"inline-block", width:8, height:8, borderRadius:"50%", background: autoMode ? "#10b981" : "#9ca3af", boxShadow: autoMode ? "0 0 0 4px rgba(16,185,129,0.18)" : "none", animation: autoMode ? "pulse 1.6s ease-in-out infinite" : "none" }} />
                Agentic AI · {autoMode ? "Autopilot" : "Manual"}
              </span>
              <button onClick={()=>setAutoMode(v=>!v)}
                style={{ fontSize:11, padding:"4px 10px", borderRadius:"var(--r-full)", border:"1.5px solid var(--border)", cursor:"pointer", background: autoMode ? "var(--blue)" : "var(--bg-soft)", color: autoMode ? "#fff" : "var(--text-2)", fontWeight:700 }}>
                {autoMode ? "ON" : "OFF"}
              </button>
            </div>
            <div className="card-body" style={{ display:"flex", flexDirection:"column", gap:10 }}>
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:8 }}>
                <div style={{ padding:"8px 10px", borderRadius:"var(--r-sm)", background:"var(--bg-soft)" }}>
                  <div style={{ fontSize:10, color:"var(--text-3)", textTransform:"uppercase", letterSpacing:"0.05em", fontWeight:700 }}>Confidence</div>
                  <div style={{ fontFamily:"'Sora',sans-serif", fontSize:18, fontWeight:800, color: confidence==null ? "var(--text-3)" : confidence>=0.85 ? "var(--green)" : confidence>=0.7 ? "var(--amber)" : "var(--red)" }}>
                    {confidence==null ? "—" : `${Math.round(confidence*100)}%`}
                  </div>
                </div>
                <div style={{ padding:"8px 10px", borderRadius:"var(--r-sm)", background:"var(--bg-soft)" }}>
                  <div style={{ fontSize:10, color:"var(--text-3)", textTransform:"uppercase", letterSpacing:"0.05em", fontWeight:700 }}>Fleet</div>
                  <div style={{ fontFamily:"'Sora',sans-serif", fontSize:18, fontWeight:800, color:"var(--text)" }}>
                    {aiStatus?.agents ?? aiStatus?.fleet?.length ?? 13}<span style={{ fontSize:11, fontWeight:500, color:"var(--text-3)", marginLeft:4 }}>agents</span>
                  </div>
                </div>
              </div>

              {/* Live pipeline visualization — 12 stages, automated indicator */}
              <div style={{ background:"var(--bg-soft)", borderRadius:"var(--r-sm)", padding:"10px 12px" }}>
                <div style={{ fontSize:10, color:"var(--text-3)", textTransform:"uppercase", letterSpacing:"0.05em", fontWeight:700, marginBottom:8, display:"flex", justifyContent:"space-between" }}>
                  <span>Autonomous Pipeline</span>
                  <span style={{ color: loading ? "var(--blue)" : "var(--green)" }}>
                    {loading ? "● running" : confidence != null ? "✓ idle" : "○ standby"}
                  </span>
                </div>
                <div style={{ display:"grid", gridTemplateColumns:`repeat(${PIPELINE.length}, 1fr)`, gap:3 }}>
                  {PIPELINE.map((p, i) => {
                    const done = i < pipelineStage;
                    const active = i === pipelineStage && loading;
                    return (
                      <div key={p.id}
                        title={p.label}
                        style={{
                          height:6, borderRadius:3,
                          background: done ? "var(--green)" : active ? "var(--blue)" : "var(--border)",
                          boxShadow: active ? "0 0 8px rgba(37,99,235,0.45)" : "none",
                          transition:"background 0.3s, box-shadow 0.3s",
                          animation: active ? "pulse 1.2s ease-in-out infinite" : "none",
                        }} />
                    );
                  })}
                </div>
                <div style={{ marginTop:6, fontSize:10.5, color:"var(--text-2)", display:"flex", justifyContent:"space-between" }}>
                  <span>{PIPELINE[pipelineStage]?.icon} {PIPELINE[pipelineStage]?.label || "Idle"}</span>
                  <span>{Math.round(((pipelineStage+1)/PIPELINE.length)*100)}%</span>
                </div>
              </div>

              {/* Parallel automated tasks queue */}
              {taskQueue.length > 0 && (
                <div style={{ background:"var(--bg-soft)", borderRadius:"var(--r-sm)", padding:"10px 12px" }}>
                  <div style={{ fontSize:10, color:"var(--text-3)", textTransform:"uppercase", letterSpacing:"0.05em", fontWeight:700, marginBottom:6 }}>
                    Automated Task Queue
                  </div>
                  <div style={{ display:"flex", flexDirection:"column", gap:4 }}>
                    {taskQueue.map(t => (
                      <div key={t.id} style={{ display:"flex", alignItems:"center", gap:8, fontSize:11.5 }}>
                        <span style={{
                          width:14, height:14, borderRadius:"50%", flexShrink:0,
                          background: t.status==="done" ? "var(--green)" : t.status==="running" ? "var(--blue)" : "var(--border)",
                          color:"#fff", fontSize:9, display:"flex", alignItems:"center", justifyContent:"center",
                          fontWeight:800,
                          animation: t.status==="running" ? "pulse 1.2s ease-in-out infinite" : "none",
                        }}>
                          {t.status==="done" ? "✓" : t.status==="running" ? "▶" : ""}
                        </span>
                        <span style={{
                          color: t.status==="done" ? "var(--text-3)" : "var(--text-2)",
                          textDecoration: t.status==="done" ? "line-through" : "none",
                          flex:1,
                        }}>
                          {t.name}
                        </span>
                        <span style={{ fontSize:10, color: t.status==="running" ? "var(--blue)" : "var(--text-3)", fontWeight:600 }}>
                          {t.status}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              <div style={{ display:"flex", gap:6 }}>
                <button onClick={()=>runReplan("user-triggered refresh")}
                  disabled={healing || !itinerary}
                  style={{ flex:1, fontSize:12, padding:"8px 10px", border:"1.5px solid var(--blue)", borderRadius:"var(--r-md)", background: healing ? "var(--bg-soft)" : "var(--blue-dim)", color:"var(--blue)", cursor:(healing||!itinerary)?"not-allowed":"pointer", fontWeight:700 }}>
                  {healing ? "Healing…" : "🔄 Auto Replan"}
                </button>
                <button onClick={()=>{ setAgentLog([]); pushLog("Log cleared","info"); }}
                  style={{ fontSize:12, padding:"8px 10px", border:"1.5px solid var(--border)", borderRadius:"var(--r-md)", background:"var(--bg-soft)", color:"var(--text-2)", cursor:"pointer", fontWeight:600 }}>
                  Clear
                </button>
              </div>
              <div style={{ borderTop:"1px solid var(--border)", paddingTop:8, maxHeight:200, overflowY:"auto", display:"flex", flexDirection:"column", gap:4 }}>
                {agentLog.length===0 ? (
                  <div style={{ fontSize:11.5, color:"var(--text-3)", textAlign:"center", padding:"8px 0" }}>
                    {autoMode ? "Autopilot armed — waiting for action…" : "Enable Autopilot to live-stream agent decisions"}
                  </div>
                ) : agentLog.map((e,i)=>(
                  <motion.div key={e.ts+"_"+i} initial={{opacity:0,x:-6}} animate={{opacity:1,x:0}}
                    style={{ fontSize:11, fontFamily:"monospace", color: e.kind==="error" ? "var(--red)" : e.kind==="warn" ? "var(--amber)" : e.kind==="success" ? "var(--green)" : "var(--text-2)", lineHeight:1.4 }}>
                    {new Date(e.ts).toLocaleTimeString().slice(0,8)} · {e.msg}
                  </motion.div>
                ))}
              </div>
              {aiStatus?.fleet?.length > 0 && (
                <details style={{ fontSize:11, color:"var(--text-2)" }}>
                  <summary style={{ cursor:"pointer", fontWeight:700, color:"var(--text)" }}>Fleet roster ({aiStatus.fleet.length})</summary>
                  <div style={{ marginTop:6, display:"flex", flexDirection:"column", gap:3 }}>
                    {aiStatus.fleet.slice(0,12).map((a,i)=>(
                      <div key={i} style={{ display:"flex", justifyContent:"space-between", gap:8 }}>
                        <span style={{ fontSize:10.5 }}>{a.name || a.role || `Agent ${i+1}`}</span>
                        <span style={{ fontSize:10, color:"var(--green)" }}>● {a.status || "online"}</span>
                      </div>
                    ))}
                  </div>
                </details>
              )}
            </div>
          </div>

          <div className="card">
            <div className="card-header"><span className="card-title">Budget Overview</span></div>
            <div className="card-body">
              <div style={{ fontFamily:"'Sora',sans-serif", fontSize:24, fontWeight:800, color:"var(--text)", letterSpacing:"-0.02em", marginBottom:14 }}>₹{form.budget.toLocaleString("en-IN")}</div>
              {[["Stay",35,"var(--purple)"],["Food",22,"var(--green)"],["Activities",18,"#db2777"],["Transit",15,"var(--amber)"]].map(([l,p,c])=>(
                <div key={l} style={{ marginBottom:10 }}>
                  <div className="budget-cat-row"><div className="budget-cat-dot" style={{background:c}}/><span className="budget-cat-label">{l}</span><span className="budget-cat-pct">{p}%</span></div>
                  <div className="progress-wrap"><div className="progress-fill" style={{width:`${p}%`,background:c}}/></div>
                </div>
              ))}
            </div>
          </div>

          {itinerary?.isSrmDestination && (
            <div className="card" style={{ background:"linear-gradient(135deg,#fef3c7,#fffbeb)", border:"1.5px solid #f59e0b" }}>
              <div className="card-header"><span className="card-title" style={{ color:"#b45309" }}>⭐ SRM Destination</span></div>
              <div className="card-body" style={{ fontSize:13, color:"var(--text-2)", lineHeight:1.6 }}>
                Your destination is an SRM campus. Premium boys/girls hostel applications and SRM-official hotel options will appear in the Hotels section. <br/>
                <a href="https://www.srmist.edu.in/hostels/" target="_blank" rel="noopener noreferrer"
                  style={{ color:"#b45309", fontWeight:700, textDecoration:"none", marginTop:6, display:"inline-block" }}>
                  Apply for Hostel →
                </a>
              </div>
            </div>
          )}

          <div className="card">
            <div className="card-header"><span className="card-title">Smart Tips</span></div>
            <div className="card-body" style={{ display:"flex", flexDirection:"column", gap:10 }}>
              {["Book hotels 2 weeks early for 15% savings","Visit attractions before 10 AM for 40% less crowd","Street food saves ₹400–800 daily vs restaurants","Group bookings unlock 10–20% transport discounts"].map((tip,i)=>(
                <div key={i} style={{ fontSize:12.5, color:"var(--text-2)", lineHeight:1.5, padding:"8px 10px", background:"var(--bg-soft)", borderRadius:"var(--r-sm)" }}>{tip}</div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
