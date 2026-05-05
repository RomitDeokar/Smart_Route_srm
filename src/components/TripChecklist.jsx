import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

const CATEGORIES = {
  documents: { label:"Documents", color:"var(--blue)" },
  clothing:  { label:"Clothing",  color:"var(--purple)" },
  tech:      { label:"Tech & Power", color:"var(--teal)" },
  health:    { label:"Health",    color:"var(--green)" },
  misc:      { label:"Misc",      color:"var(--amber)" },
};

const SMART_ITEMS = {
  documents: ["Government ID (Aadhaar/Passport)", "Hotel booking confirmation", "Flight e-tickets", "Travel insurance copy", "Emergency contact card"],
  clothing:  ["Walking shoes", "Light layers (3 sets)", "Rain jacket / windbreaker", "Comfortable sleepwear", "Warm socks"],
  tech:      ["Phone charger + cable", "Power bank (20000 mAh)", "Universal adapter", "Earphones", "Camera or spare SD card"],
  health:    ["Personal medications", "First aid kit (bandages, antiseptic)", "ORS sachets", "Sunscreen SPF50+", "Insect repellent"],
  misc:      ["Reusable water bottle", "Snacks for journey", "Small day backpack", "Padlock for hostel", "Cash (₹2000 emergency)"],
};

export default function TripChecklist({ destination, weather, persona, days, addToast }) {
  const [items, setItems]     = useState([]);
  const [checked, setChecked] = useState({});
  const [loading, setLoading] = useState(false);
  const [custom, setCustom]   = useState("");
  const [filter, setFilter]   = useState("all");

  const defaultItems = () => {
    const flat = Object.entries(SMART_ITEMS).flatMap(([cat, list]) =>
      list.map((name, i) => ({ id: `${cat}-${i}`, name, category: cat }))
    );
    setItems(flat);
  };

  useEffect(() => { defaultItems(); }, []);

  const fetchAIList = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/packing-list", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ destination, travel_dates: `${days} days`, persona, days }),
      });
      const data = await res.json();
      if (data.ok && data.items?.length) {
        const aiItems = data.items.map((name, i) => ({
          id: `ai-${i}`, name, category: guessCategory(name),
        }));
        setItems(prev => {
          const existing = new Set(prev.map(p=>p.name.toLowerCase()));
          const fresh = aiItems.filter(a => !existing.has(a.name.toLowerCase()));
          return [...prev, ...fresh];
        });
        addToast?.(`${data.items.length} AI packing items generated!`, "success");
      }
    } catch(e) { addToast?.(e.message, "error"); }
    finally { setLoading(false); }
  };

  const guessCategory = (name) => {
    const n = name.toLowerCase();
    if (/id|passport|ticket|booking|insurance|document|card/.test(n)) return "documents";
    if (/shoe|cloth|jacket|shirt|pant|sock|wear|layer|outfit/.test(n)) return "clothing";
    if (/charger|power|bank|phone|camera|adapter|earphone|tech|cable/.test(n)) return "tech";
    if (/medicine|first aid|sunscreen|repellent|health|kit|sachet|ors/.test(n)) return "health";
    return "misc";
  };

  const toggle = (id) => setChecked(c => ({ ...c, [id]: !c[id] }));

  const addCustom = () => {
    if (!custom.trim()) return;
    setItems(prev => [...prev, { id: `custom-${Date.now()}`, name: custom.trim(), category: guessCategory(custom) }]);
    setCustom("");
  };

  const removeItem = (id) => setItems(prev => prev.filter(i => i.id !== id));

  const filteredItems = filter === "all" ? items : items.filter(i => i.category === filter);
  const doneCount = items.filter(i => checked[i.id]).length;
  const pct = items.length > 0 ? Math.round((doneCount / items.length) * 100) : 0;

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="9 11 12 14 22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>
          Trip Checklist
        </span>
        <div style={{ display:"flex", alignItems:"center", gap:8 }}>
          <span style={{ fontSize:12, color:"var(--text-2)" }}>{doneCount}/{items.length}</span>
          <span className="pill green" style={{ fontSize:10 }}>{pct}% packed</span>
        </div>
      </div>

      <div style={{ padding:"10px 20px 0" }}>
        {/* Progress */}
        <div className="progress-wrap" style={{ marginBottom:12 }}>
          <motion.div className="progress-fill" style={{ background:"var(--g-blue)" }} initial={{ width:0 }} animate={{ width:`${pct}%` }} transition={{ duration:0.5 }} />
        </div>

        {/* Category filter */}
        <div style={{ display:"flex", gap:6, flexWrap:"wrap", marginBottom:12 }}>
          <button className={filter==="all"?"chip active":"chip"} onClick={()=>setFilter("all")}>All</button>
          {Object.entries(CATEGORIES).map(([k,v])=>(
            <button key={k} className={filter===k?"chip active":"chip"} onClick={()=>setFilter(k)}
              style={filter===k?{background:`${v.color}15`,borderColor:v.color,color:v.color}:{}}
            >
              {v.label}
            </button>
          ))}
        </div>

        {/* AI generate */}
        <button className="btn btn-ghost btn-sm w-full" onClick={fetchAIList} disabled={loading} style={{ marginBottom:12, display:"flex", alignItems:"center", justifyContent:"center", gap:7 }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
          {loading ? "AI is generating list..." : `Generate AI list for ${destination||"your trip"}`}
        </button>

        {/* Add custom */}
        <div style={{ display:"flex", gap:8, marginBottom:14 }}>
          <input className="field-input" placeholder="Add custom item..." value={custom} onChange={e=>setCustom(e.target.value)} onKeyDown={e=>e.key==="Enter"&&addCustom()} style={{ fontSize:13 }} />
          <button className="btn btn-primary btn-sm" onClick={addCustom} style={{ flexShrink:0 }}>Add</button>
        </div>
      </div>

      <div className="checklist" style={{ padding:"0 20px 16px", maxHeight:320, overflowY:"auto" }}>
        <AnimatePresence>
          {filteredItems.map(item => {
            const cat = CATEGORIES[item.category];
            return (
              <motion.div
                key={item.id}
                layout
                initial={{ opacity:0, x:-8 }}
                animate={{ opacity:1, x:0 }}
                exit={{ opacity:0, x:8, height:0 }}
                className={`check-item ${checked[item.id]?"checked":""}`}
                onClick={()=>toggle(item.id)}
              >
                <div className="check-box" style={checked[item.id]?{}:{}}>
                  {checked[item.id] && (
                    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                  )}
                </div>
                <span className="check-label">{item.name}</span>
                <span className="check-cat" style={{ color: cat?.color }}>{cat?.label}</span>
                <button
                  onClick={e=>{ e.stopPropagation(); removeItem(item.id); }}
                  style={{ background:"none", color:"var(--text-3)", cursor:"pointer", padding:"0 4px", fontSize:16, lineHeight:1 }}
                >×</button>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>

      {pct === 100 && items.length > 0 && (
        <div style={{ margin:"0 20px 16px", padding:"10px 14px", background:"var(--green-bg)", border:"1.5px solid rgba(22,163,74,0.2)", borderRadius:"var(--r-md)", fontSize:13, color:"var(--green)", fontWeight:600, textAlign:"center" }}>
          All packed! Have a great trip to {destination}!
        </div>
      )}
    </div>
  );
}
