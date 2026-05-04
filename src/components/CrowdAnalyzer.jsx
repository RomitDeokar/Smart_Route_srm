import { useState } from "react";
import { motion } from "framer-motion";

const CROWD_COLORS = { Low:"var(--green)", Medium:"var(--amber)", High:"var(--red)" };
const CROWD_BG     = { Low:"crowd-low", Medium:"crowd-medium", High:"crowd-high" };

function HourlyChart({ pattern }) {
  if (!pattern?.length) return null;
  const max = Math.max(...pattern.map(p => p.density));
  return (
    <div className="hourly-bars">
      {pattern.map((p, i) => {
        const h = Math.max(4, Math.round((p.density / (max||1)) * 38));
        const c = p.density > 0.7 ? "var(--red)" : p.density > 0.4 ? "var(--amber)" : "var(--green)";
        return (
          <div key={i} className="hour-bar-wrap">
            <div className="hour-bar" style={{ height: h, background: c, opacity: 0.8 }} />
            <div className="hour-label">{p.hour?.split(":")[0]}</div>
          </div>
        );
      })}
    </div>
  );
}

export default function CrowdAnalyzer({ destination, attractions = [], addToast }) {
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(false);
  const [customPlace, setCustomPlace] = useState("");

  const fetch_ = async () => {
    setLoading(true);
    try {
      const targets = attractions.length ? attractions : [destination, `${destination} Market`, `${destination} Fort`];
      const res = await fetch("/api/crowd-info", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ destination, attractions: targets }),
      });
      const d = await res.json();
      if (d.ok) { setData(d.locations); addToast?.("Crowd data loaded", "success"); }
    } catch(e) { addToast?.(e.message, "error"); }
    finally { setLoading(false); }
  };

  const addPlace = () => {
    if (customPlace.trim()) {
      attractions.push(customPlace.trim());
      setCustomPlace("");
      fetch_();
    }
  };

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
          Crowd Analyzer
        </span>
        <button className="btn btn-primary btn-sm" onClick={fetch_} disabled={loading}>
          {loading ? "Analyzing..." : "Analyze Now"}
        </button>
      </div>
      <div className="card-body">
        {/* Add custom place */}
        <div style={{ display:"flex", gap:8, marginBottom:14 }}>
          <input className="field-input" placeholder="Add attraction to analyze..." value={customPlace} onChange={e=>setCustomPlace(e.target.value)} onKeyDown={e=>e.key==="Enter"&&addPlace()} />
          <button className="btn btn-ghost btn-sm" onClick={addPlace}>Add</button>
        </div>

        {!data ? (
          <div style={{ textAlign:"center", padding:"20px", color:"var(--text-2)", fontSize:13 }}>
            Click "Analyze Now" to see live crowd predictions for {destination}
          </div>
        ) : (
          <div className="crowd-grid">
            {data.map((loc, i) => {
              const lvl   = loc.currentDensity > 65 ? "High" : loc.currentDensity > 40 ? "Medium" : "Low";
              const fillPct = loc.currentDensity || 45;
              return (
                <motion.div key={i} className="crowd-card" initial={{ opacity:0, y:8 }} animate={{ opacity:1, y:0 }} transition={{ delay:i*0.08 }}>
                  <div className="crowd-place">{loc.name}</div>
                  <div className="crowd-meter">
                    <div style={{ height:"100%", width:`${fillPct}%`, background: CROWD_COLORS[lvl], borderRadius:"var(--r-full)", transition:"width 0.6s ease" }}/>
                  </div>
                  <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between" }}>
                    <span className={`crowd-level ${CROWD_BG[lvl]}`}>{lvl}</span>
                    <span style={{ fontSize:11, color:"var(--text-3)" }}>{fillPct}% capacity</span>
                  </div>
                  <div className="crowd-best-time">Best time: {loc.recommended_time}</div>
                  <div style={{ fontSize:11, color:"var(--text-3)", marginTop:3 }}>Peak: {loc.peak_hours}</div>
                  {/* Mock hourly pattern */}
                  <HourlyChart pattern={Array.from({length:10},(_,h)=>({ hour:`${h+8}:00`, density: Math.min(1, 0.2 + (h>2&&h<7?0.6:0.2) + Math.random()*0.2) }))} />
                </motion.div>
              );
            })}
          </div>
        )}

        {data && (
          <div style={{ marginTop:14, padding:"10px 12px", background:"var(--bg-soft)", borderRadius:"var(--r-md)", border:"1.5px solid var(--border)", fontSize:12.5, color:"var(--text-2)" }}>
            <strong style={{ color:"var(--text)", display:"block", marginBottom:4 }}>Smart Tips</strong>
            Arrive before 10:00 AM to avoid peak crowds. Monday–Wednesday typically 30% less crowded than weekends.
            Crowd multiplier on weekends: {data[0]?.weekendMultiplier || 1.4}×.
          </div>
        )}
      </div>
    </div>
  );
}
