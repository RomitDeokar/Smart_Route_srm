import { useState, useEffect, useMemo } from "react";
import { motion } from "framer-motion";

/* Smart Packing — categorised checklist with progress bar.
   Auto-fetches a recommended list from /api/packing-list given trip context,
   merges with the user's checked-state from localStorage. */

const STORE_KEY = "sr_packing_state";

const FALLBACK_LIST = {
  "Documents":   ["ID / Aadhaar / Passport","Booking confirmations","Travel insurance","Emergency contacts card"],
  "Clothing":    ["3-4 T-shirts","2 pairs of pants","1 light jacket","Undergarments","Sleepwear","Socks (4 pairs)"],
  "Toiletries":  ["Toothbrush + paste","Shampoo / soap","Sunscreen SPF 50","Deodorant","Sanitary supplies"],
  "Electronics": ["Phone + charger","Power bank (10000 mAh)","Universal adapter","Headphones","Camera"],
  "Health":      ["Basic first-aid kit","Pain relievers","Anti-allergy meds","Hand sanitizer","N95 mask"],
  "Misc":        ["Reusable water bottle","Day backpack","Sunglasses","Cap / hat","Travel pillow"],
};

const CATEGORY_EMOJI = {
  "Documents":"📄","Clothing":"👕","Toiletries":"🧴","Electronics":"🔌",
  "Health":"💊","Misc":"🧳","Footwear":"👟","Cold weather":"🧥",
  "Rain gear":"☔","Beach":"🏖️","Trekking":"🥾",
};

export default function Packing({ tripCtx, addToast }) {
  const [list, setList]       = useState(FALLBACK_LIST);
  const [checked, setChecked] = useState(() => {
    try { return JSON.parse(localStorage.getItem(STORE_KEY)) || {}; } catch { return {}; }
  });
  const [loading, setLoading] = useState(false);

  // Persist checked state
  useEffect(() => {
    try { localStorage.setItem(STORE_KEY, JSON.stringify(checked)); } catch {}
  }, [checked]);

  // Auto-fetch a tailored list from the API
  useEffect(() => {
    let mounted = true;
    (async () => {
      if (!tripCtx?.destination) return;
      setLoading(true);
      try {
        const r = await fetch("/api/packing-list", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            destination: tripCtx.destination,
            days:        tripCtx.days,
            persona:     tripCtx.persona,
            interests:   tripCtx.services,
          }),
        });
        const d = await r.json();
        if (mounted && d?.ok && d.packingList && Object.keys(d.packingList).length) {
          setList(d.packingList);
        }
      } catch {}
      finally { if (mounted) setLoading(false); }
    })();
    return () => { mounted = false; };
  }, [tripCtx?.destination, tripCtx?.days, tripCtx?.persona]);

  const allItems = useMemo(() => Object.values(list).flat(), [list]);
  const checkedCount = allItems.filter(it => checked[it]).length;
  const pct = allItems.length ? Math.round((checkedCount / allItems.length) * 100) : 0;

  const toggle = (item) => setChecked(c => ({ ...c, [item]: !c[item] }));
  const checkAll = () => {
    const next = {}; allItems.forEach(it => { next[it] = true; }); setChecked(next);
    addToast("Marked all packed ✅", "success");
  };
  const reset = () => {
    if (!confirm("Clear all checked items?")) return;
    setChecked({}); addToast("Packing list reset", "info");
  };

  return (
    <div>
      <div className="section-title">Smart Packing</div>
      <div className="section-sub">
        AI-tailored to your trip · {tripCtx?.destination || "destination"} · {tripCtx?.days || 3} days · {tripCtx?.persona || "explorer"}
      </div>

      {/* Progress card */}
      <div className="card" style={{ marginBottom:18 }}>
        <div className="card-body">
          <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", flexWrap:"wrap", gap:8, marginBottom:10 }}>
            <div>
              <div className="eyebrow">Packing progress</div>
              <div style={{ fontFamily:"'Sora',sans-serif", fontSize:28, fontWeight:800, color:"var(--text)" }}>
                {pct}% <span style={{ fontSize:14, fontWeight:500, color:"var(--text-3)", marginLeft:8 }}>{checkedCount} / {allItems.length} items</span>
              </div>
            </div>
            <div style={{ display:"flex", gap:6 }}>
              <button className="btn btn-primary btn-sm" onClick={checkAll}>Mark all packed</button>
              <button className="btn btn-ghost btn-sm" onClick={reset}>Reset</button>
            </div>
          </div>
          <div className="progress-wrap" style={{ height:8 }}>
            <motion.div className="progress-fill"
              initial={{ width: 0 }}
              animate={{ width: `${pct}%` }}
              transition={{ duration:0.6, ease:"easeOut" }}
              style={{ background: pct === 100 ? "var(--green)" : "var(--g-blue)" }}/>
          </div>
          {loading && <div style={{ fontSize:11.5, color:"var(--text-3)", marginTop:8 }}>Fetching tailored list…</div>}
        </div>
      </div>

      {/* Categories grid */}
      <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fill,minmax(260px,1fr))", gap:14 }}>
        {Object.entries(list).map(([cat, items], idx) => {
          const catChecked = items.filter(it => checked[it]).length;
          const catPct = items.length ? Math.round((catChecked / items.length) * 100) : 0;
          return (
            <motion.div key={cat} initial={{ opacity:0, y:8 }} animate={{ opacity:1, y:0 }} transition={{ delay: idx*0.04 }}
              className="card">
              <div className="card-header">
                <span className="card-title">
                  <span style={{ fontSize:18 }}>{CATEGORY_EMOJI[cat] || "📦"}</span> {cat}
                </span>
                <span className="pill blue">{catChecked} / {items.length}</span>
              </div>
              <div className="card-body" style={{ display:"flex", flexDirection:"column", gap:8 }}>
                <div className="progress-wrap" style={{ height:4 }}>
                  <div className="progress-fill" style={{ width: `${catPct}%`, background: catPct === 100 ? "var(--green)" : "var(--blue)" }} />
                </div>
                {items.map((it, i) => {
                  const isOn = !!checked[it];
                  return (
                    <label key={i} style={{
                      display:"flex", alignItems:"center", gap:8, padding:"7px 8px",
                      borderRadius:"var(--r-sm)", cursor:"pointer",
                      background: isOn ? "var(--green-bg)" : "transparent",
                      border:"1px solid", borderColor: isOn ? "rgba(22,163,74,0.18)" : "var(--border)",
                      transition:"all 0.12s",
                    }}>
                      <input type="checkbox" checked={isOn} onChange={() => toggle(it)}
                        style={{ width:16, height:16, accentColor:"var(--blue)", cursor:"pointer", flexShrink:0 }} />
                      <span style={{
                        fontSize:13, color: isOn ? "var(--text-3)" : "var(--text-2)",
                        textDecoration: isOn ? "line-through" : "none",
                      }}>{it}</span>
                    </label>
                  );
                })}
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
