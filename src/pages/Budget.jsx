import { useState, useMemo, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

const CATS = [
  { id:"Hotels",     label:"Hotels & Stay",  color:"#7c3aed", pct:0.30 },
  { id:"Food",       label:"Food & Dining",  color:"#16a34a", pct:0.20 },
  { id:"Transport",  label:"Transport",      color:"#d97706", pct:0.20 },
  { id:"Activities", label:"Activities",     color:"#db2777", pct:0.15 },
  { id:"Shopping",   label:"Shopping",       color:"#2563eb", pct:0.15 },
];

const STYPE_COLOR = { danger:"var(--red)", warning:"var(--amber)", suggestion:"var(--blue)", success:"var(--green)", info:"var(--text-2)" };
const STYPE_BG    = { danger:"var(--red-bg)", warning:"var(--amber-bg)", suggestion:"var(--blue-dim)", success:"var(--green-bg)", info:"var(--bg-soft)" };

const fmt = n => `₹${Number(n || 0).toLocaleString("en-IN")}`;

function SvgDonut({ cats, budget }) {
  const total = cats.reduce((s, c) => s + (c.allocated || 0), 0) || 1;
  const R = 70, CX = 90, CY = 90, SW = 20, circ = 2 * Math.PI * R;
  let cursor = -0.25;
  return (
    <svg width="180" height="180" viewBox="0 0 180 180">
      <circle cx={CX} cy={CY} r={R} fill="none" stroke="#f0f2f8" strokeWidth={SW}/>
      {cats.map(cat => {
        const frac  = (cat.allocated || 0) / total;
        const dash  = circ * frac - 2;
        const gap   = circ - dash;
        const offset = -(cursor * circ);
        cursor += frac;
        return (
          <circle key={cat.id} cx={CX} cy={CY} r={R} fill="none"
            stroke={cat.color} strokeWidth={SW}
            strokeDasharray={`${Math.max(0, dash)} ${gap}`}
            strokeDashoffset={offset} strokeLinecap="round"
            style={{ transition: "stroke-dasharray 0.6s ease" }}
          />
        );
      })}
      <text x={CX} y={CY - 8} textAnchor="middle" fill="#111827" fontSize="16" fontWeight="800" fontFamily="'Sora',sans-serif">
        {fmt(budget)}
      </text>
      <text x={CX} y={CY + 10} textAnchor="middle" fill="#6b7280" fontSize="11" fontFamily="'Plus Jakarta Sans',sans-serif">
        Total budget
      </text>
    </svg>
  );
}

export default function Budget({ tripCtx, addToast }) {
  const [budget, setBudget]   = useState(tripCtx.budget || 45800);
  const [status, setStatus]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [expense, setExpense] = useState({ category: "Hotels", amount: 500 });
  const [history, setHistory] = useState([]);
  const [suggest, setSuggest] = useState([]);

  const cats = useMemo(() => CATS.map(c => ({
    ...c,
    allocated: status ? (status.categories?.[c.id]?.allocated || Math.round(budget * c.pct)) : Math.round(budget * c.pct),
    spent:     status?.categories?.[c.id]?.spent    || 0,
    progress:  status?.categories?.[c.id]?.progress || 0,
  })), [budget, status]);

  const totalSpent  = status?.totalSpent || 0;
  const healthPct   = status ? Math.round((totalSpent / budget) * 100) : 0;
  const healthColor = healthPct < 60 ? "var(--green)" : healthPct < 85 ? "var(--amber)" : "var(--red)";

  const getSuggestions = async () => {
    try {
      const res = await fetch("/api/budget/suggest");
      const data = await res.json();
      if (data.ok) setSuggest(data.suggestions || []);
    } catch {}
  };

  useEffect(() => { if (status) getSuggestions(); }, [status]);

  const createBudget = async () => {
    setLoading(true);
    try {
      const allocs = CATS.reduce((o, c) => ({ ...o, [c.id]: Math.round(budget * c.pct) }), {});
      const res  = await fetch("/api/budget/create", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ total_budget: budget, allocations: allocs }) });
      const data = await res.json();
      if (!data.ok) throw new Error(data.error);
      setStatus(data.budget);
      addToast(`Budget of ${fmt(budget)} created!`, "success");
    } catch (e) { addToast(e.message, "error"); }
    finally { setLoading(false); }
  };

  const addExpense = async () => {
    if (!status) { addToast("Create a budget first", "warning"); return; }
    setLoading(true);
    try {
      const res  = await fetch("/api/budget/update", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(expense) });
      const data = await res.json();
      if (!data.ok) throw new Error(data.error);
      setStatus(data.budget);
      setHistory(h => [{ ...expense, at: new Date().toLocaleTimeString() }, ...h].slice(0, 10));
      addToast(`${fmt(expense.amount)} added to ${expense.category}`, "success");
    } catch (e) { addToast(e.message, "error"); }
    finally { setLoading(false); }
  };

  return (
    <div>
      <div className="section-title">Budget Manager</div>
      <div className="section-sub">Track spending across all categories with AI suggestions</div>

      {/* Stats */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 14, marginBottom: 22 }}>
        {[
          { label: "Total Budget",  value: fmt(budget),               accent: "var(--blue)" },
          { label: "Total Spent",   value: fmt(totalSpent),           accent: healthColor, sub: `${healthPct}% used` },
          { label: "Remaining",     value: fmt(budget - totalSpent),  accent: "var(--green)" },
          { label: "Health",        value: healthPct < 60 ? "Healthy" : healthPct < 85 ? "Monitor" : "Exceeded", accent: healthColor },
        ].map(s => (
          <div key={s.label} className="stat-card">
            <div className="stat-label">{s.label}</div>
            <div className="stat-value" style={{ color: s.accent, fontSize: 20 }}>{s.value}</div>
            {s.sub && <div className="stat-sub">{s.sub}</div>}
          </div>
        ))}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 18, marginBottom: 18 }}>
        {/* Donut */}
        <div className="card">
          <div className="card-header"><span className="card-title">Allocation Breakdown</span></div>
          <div className="card-body" style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 16 }}>
            <SvgDonut cats={cats} budget={budget}/>
            <div style={{ width: "100%", display: "flex", flexDirection: "column", gap: 8 }}>
              {cats.map(cat => (
                <div key={cat.id} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <div style={{ width: 10, height: 10, borderRadius: "50%", background: cat.color, flexShrink: 0 }}/>
                  <span style={{ fontSize: 13, color: "var(--text-2)", flex: 1 }}>{cat.label}</span>
                  <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text)" }}>{fmt(cat.allocated)}</span>
                  <span style={{ fontSize: 12, color: "var(--text-3)", minWidth: 30, textAlign: "right" }}>{Math.round(cat.pct * 100)}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Progress bars */}
        <div className="card">
          <div className="card-header"><span className="card-title">Spending Progress</span></div>
          <div className="card-body" style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {cats.map(cat => (
              <div key={cat.id}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                  <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text)" }}>{cat.id}</span>
                  <span style={{ fontSize: 12, color: "var(--text-3)" }}>{fmt(cat.spent)} / {fmt(cat.allocated)}</span>
                </div>
                <div className="progress-wrap">
                  <motion.div className="progress-fill" style={{ background: cat.color }}
                    initial={{ width: 0 }} animate={{ width: `${Math.min(cat.progress || 0, 100)}%` }}
                    transition={{ duration: 0.7 }}/>
                </div>
                <div style={{ fontSize: 11, color: "var(--text-3)", marginTop: 3 }}>
                  {cat.progress || 0}% used
                  {(cat.progress || 0) > 85 && <span style={{ color: "var(--red)", marginLeft: 8, fontWeight: 600 }}>Near limit</span>}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 18 }}>
        {/* Set budget */}
        <div className="card">
          <div className="card-header"><span className="card-title">Set Budget</span></div>
          <div className="card-body" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <div className="field-group">
              <label className="field-label">Total Budget (₹)</label>
              <input type="number" className="field-input" step={500} value={budget} onChange={e => setBudget(+e.target.value)}/>
            </div>
            <input type="range" min={5000} max={200000} step={1000} value={budget}
              onChange={e => setBudget(+e.target.value)} style={{ width: "100%", accentColor: "var(--blue)" }}/>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11.5, color: "var(--text-3)" }}>
              <span>₹5K</span><span style={{ fontWeight: 600, color: "var(--blue)" }}>{fmt(budget)}</span><span>₹2L</span>
            </div>
            <button className="btn btn-primary w-full" onClick={createBudget} disabled={loading}>
              {loading ? "Saving..." : "Create Budget"}
            </button>
          </div>
        </div>

        {/* Log expense */}
        <div className="card">
          <div className="card-header"><span className="card-title">Log Expense</span></div>
          <div className="card-body" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <div className="field-group">
              <label className="field-label">Category</label>
              <select className="field-input" value={expense.category} onChange={e => setExpense(x => ({ ...x, category: e.target.value }))}>
                {CATS.map(c => <option key={c.id} value={c.id}>{c.id}</option>)}
              </select>
            </div>
            <div className="field-group">
              <label className="field-label">Amount (₹)</label>
              <input type="number" className="field-input" step={50} value={expense.amount}
                onChange={e => setExpense(x => ({ ...x, amount: +e.target.value }))}/>
            </div>
            <button className="btn btn-primary w-full" onClick={addExpense} disabled={loading}
              style={{ background: "var(--g-green)" }}>
              {loading ? "Adding..." : "+ Log Expense"}
            </button>
          </div>
        </div>

        {/* AI suggestions */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">AI Suggestions</span>
            <button className="btn btn-ghost btn-sm" onClick={getSuggestions}>Refresh</button>
          </div>
          <div className="card-body" style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {suggest.length === 0
              ? <p style={{ fontSize: 13, color: "var(--text-3)" }}>Create a budget to get AI suggestions.</p>
              : suggest.map((s, i) => (
                <motion.div key={i} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.06 }}
                  style={{ padding: "9px 12px", borderRadius: "var(--r-md)", fontSize: 12.5, lineHeight: 1.5,
                    background: STYPE_BG[s.type] || "var(--bg-soft)", color: STYPE_COLOR[s.type] || "var(--text-2)",
                    border: `1.5px solid ${STYPE_COLOR[s.type] || "var(--border)"}30` }}>
                  {s.text}
                </motion.div>
              ))
            }
          </div>
        </div>
      </div>

      {/* History */}
      <AnimatePresence>
        {history.length > 0 && (
          <motion.div className="card" style={{ marginTop: 18 }} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
            <div className="card-header"><span className="card-title">Expense History</span></div>
            <div className="card-body">
              {history.map((h, i) => (
                <div key={i} style={{ display: "flex", alignItems: "center", gap: 12, padding: "8px 10px",
                  background: "var(--bg-soft)", borderRadius: "var(--r-sm)", border: "1.5px solid var(--border)",
                  fontSize: 13, marginBottom: 6 }}>
                  <span style={{ color: "var(--text-3)", minWidth: 70, fontSize: 11.5 }}>{h.at}</span>
                  <span style={{ fontWeight: 600, color: "var(--text)", flex: 1 }}>{h.category}</span>
                  <span style={{ fontWeight: 700, color: "var(--blue)" }}>{fmt(h.amount)}</span>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
