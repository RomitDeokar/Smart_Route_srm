import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import PayButton from "../components/PayButton.jsx";

const SEED = [
  { id:1, type:"flight",   name:"Chennai → Shillong",       sub:"IndiGo 6E-234 · Apr 28 · 06:45 AM",   price:4200, status:"confirmed" },
  { id:2, type:"hotel",    name:"The Grand Regency",         sub:"Deluxe Suite · Apr 28–30 · 2 Nights",  price:8499, status:"confirmed" },
  { id:3, type:"activity", name:"Ward's Lake Heritage Walk", sub:"Apr 29 · 9:00 AM · 2 persons",         price:600,  status:"pending"   },
  { id:4, type:"flight",   name:"Shillong → Chennai",       sub:"Air India AI-512 · May 2 · 08:15 AM",  price:3800, status:"confirmed" },
  { id:5, type:"hotel",    name:"Urban Transit VIP Van",     sub:"Airport Drop · May 2 · 6:00 AM",       price:1250, status:"pending"   },
  { id:6, type:"activity", name:"Elephant Falls Day Tour",   sub:"Apr 29 · 2:00 PM · Guide included",    price:450,  status:"cancelled" },
];

const ICON_STYLES = {
  flight:   { bg:"#eff6ff", color:"#2563eb" },
  hotel:    { bg:"#f5f3ff", color:"#7c3aed" },
  activity: { bg:"#f0fdf4", color:"#16a34a" },
};

const STATUS_STYLE = {
  confirmed: { cls:"green",  label:"Confirmed" },
  pending:   { cls:"amber",  label:"Pending"   },
  cancelled: { cls:"red",    label:"Cancelled" },
};

const fmt = n => `₹${Number(n || 0).toLocaleString("en-IN")}`;

function TypeIcon({ type }) {
  if (type === "flight") return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 16v-2l-8-5V3.5a1.5 1.5 0 0 0-3 0V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5z"/>
    </svg>
  );
  if (type === "hotel") return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
      <polyline points="9 22 9 12 15 12 15 22"/>
    </svg>
  );
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/>
      <circle cx="12" cy="10" r="3"/>
    </svg>
  );
}

export default function Reservations({ addToast }) {
  const [bookings, setBookings] = useState(SEED);
  const [tab,  setTab]  = useState("All");
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState({ type:"flight", name:"", sub:"", price:"" });

  const TABS = ["All","Flights","Hotels","Activities"];
  const filtered = tab === "All" ? bookings : bookings.filter(r => r.type === tab.toLowerCase().replace(/s$/, ""));
  const total = bookings.filter(r => r.status !== "cancelled").reduce((s, r) => s + r.price, 0);

  const save = () => {
    if (!form.name.trim()) { addToast("Name is required","warning"); return; }
    setBookings(b => [{ id: Date.now(), ...form, price: +form.price || 0, status: "pending" }, ...b]);
    setOpen(false);
    setForm({ type:"flight", name:"", sub:"", price:"" });
    addToast("Booking saved!", "success");
  };

  const cancel = (id) => {
    setBookings(b => b.map(r => r.id === id ? { ...r, status:"cancelled" } : r));
    addToast("Booking cancelled", "info");
  };

  const confirm = (id, name) => {
    setBookings(b => b.map(r => r.id === id ? { ...r, status:"confirmed" } : r));
    addToast(`${name} confirmed!`, "success");
  };

  return (
    <div>
      <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:4 }}>
        <div>
          <div className="section-title">Reservations</div>
          <div className="section-sub">All your bookings in one place</div>
        </div>
        <button className="btn btn-primary" onClick={() => setOpen(true)}>+ New Booking</button>
      </div>

      {/* Stats */}
      <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:14, marginBottom:22 }}>
        {[
          { label:"Total Bookings", value:bookings.length },
          { label:"Confirmed",      value:bookings.filter(r=>r.status==="confirmed").length, accent:"var(--green)" },
          { label:"Pending",        value:bookings.filter(r=>r.status==="pending").length,   accent:"var(--amber)" },
          { label:"Total Value",    value:fmt(total) },
        ].map(s => (
          <div key={s.label} className="stat-card">
            <div className="stat-label">{s.label}</div>
            <div className="stat-value" style={{ fontSize:22, ...(s.accent ? { color:s.accent }:{}) }}>{s.value}</div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <div style={{ display:"flex", gap:6, marginBottom:16 }}>
        {TABS.map(t => <button key={t} className={tab===t?"chip active":"chip"} onClick={()=>setTab(t)}>{t}</button>)}
      </div>

      {/* List */}
      <div className="reservations-list">
        <AnimatePresence>
          {filtered.map((r, i) => {
            const ic   = ICON_STYLES[r.type] || ICON_STYLES.activity;
            const stat = STATUS_STYLE[r.status] || STATUS_STYLE.pending;
            return (
              <motion.div key={r.id} className="reservation-row"
                initial={{opacity:0,x:-14}} animate={{opacity:1,x:0}} transition={{delay:i*0.05}}>
                <div className="res-icon" style={{ background:ic.bg, color:ic.color, border:`1.5px solid ${ic.color}20` }}>
                  <TypeIcon type={r.type}/>
                </div>
                <div style={{ flex:1, minWidth:0 }}>
                  <div className="res-name">{r.name}</div>
                  <div className="res-sub">{r.sub}</div>
                </div>
                <div style={{ textAlign:"right", flexShrink:0 }}>
                  <div className="res-price">{fmt(r.price)}</div>
                  <span className={`pill ${stat.cls}`} style={{ marginTop:5, display:"inline-flex" }}>{stat.label}</span>
                </div>
                {r.status !== "cancelled" && (
                  <div style={{ display:"flex", gap:6, flexShrink:0 }}>
                    <PayButton
                      items={[{ name:r.name, price:r.price, quantity:1 }]}
                      metadata={{ type:r.type }}
                      label="Pay"
                      style={{ padding:"6px 14px", fontSize:12.5, background:"var(--g-green)" }}
                      addToast={addToast}
                      onSuccess={() => confirm(r.id, r.name)}
                    />
                    <button className="btn btn-ghost btn-sm" onClick={()=>cancel(r.id)}
                      style={{ color:"var(--red)", border:"1.5px solid rgba(220,38,38,0.2)" }}>
                      Cancel
                    </button>
                  </div>
                )}
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>

      {/* Modal */}
      <AnimatePresence>
        {open && (
          <div style={{ position:"fixed", inset:0, background:"rgba(0,0,0,0.45)", backdropFilter:"blur(6px)", zIndex:999, display:"flex", alignItems:"center", justifyContent:"center", padding:16 }}>
            <motion.div initial={{scale:0.92,opacity:0}} animate={{scale:1,opacity:1}} exit={{scale:0.95,opacity:0}}
              style={{ background:"white", borderRadius:"var(--r-2xl)", width:"100%", maxWidth:420, padding:28, boxShadow:"0 24px 64px rgba(0,0,0,0.15)" }}>
              <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:20 }}>
                <div style={{ fontFamily:"'Sora',sans-serif", fontSize:17, fontWeight:700 }}>New Booking</div>
                <button onClick={()=>setOpen(false)} style={{ background:"none", cursor:"pointer", color:"var(--text-3)", fontSize:20 }}>×</button>
              </div>
              <div style={{ display:"flex", flexDirection:"column", gap:14 }}>
                <div className="field-group">
                  <label className="field-label">Type</label>
                  <select className="field-input" value={form.type} onChange={e=>setForm(f=>({...f,type:e.target.value}))}>
                    <option value="flight">Flight</option>
                    <option value="hotel">Hotel</option>
                    <option value="activity">Activity</option>
                  </select>
                </div>
                <div className="field-group">
                  <label className="field-label">Name / Route</label>
                  <input className="field-input" placeholder="e.g. Chennai → Goa" value={form.name} onChange={e=>setForm(f=>({...f,name:e.target.value}))}/>
                </div>
                <div className="field-group">
                  <label className="field-label">Details</label>
                  <input className="field-input" placeholder="Date, flight code, notes..." value={form.sub} onChange={e=>setForm(f=>({...f,sub:e.target.value}))}/>
                </div>
                <div className="field-group">
                  <label className="field-label">Amount (₹)</label>
                  <input className="field-input" type="number" placeholder="0" value={form.price} onChange={e=>setForm(f=>({...f,price:e.target.value}))}/>
                </div>
                <div style={{ display:"flex", gap:8 }}>
                  <button className="btn btn-primary" style={{ flex:1 }} onClick={save}>Save Booking</button>
                  <button className="btn btn-ghost" onClick={()=>setOpen(false)}>Cancel</button>
                </div>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
