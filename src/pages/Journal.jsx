import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

const MOODS = [
  { id: "happy",     label: "Happy",     emoji: "😊", color: "#fbbf24" },
  { id: "amazed",    label: "Amazed",    emoji: "🤩", color: "#a78bfa" },
  { id: "relaxed",   label: "Relaxed",   emoji: "😌", color: "#34d399" },
  { id: "adventure", label: "Adventure", emoji: "🤠", color: "#f97316" },
  { id: "tired",     label: "Tired",     emoji: "😴", color: "#94a3b8" },
  { id: "wow",       label: "Wow!",      emoji: "🥳", color: "#ec4899" },
];

export default function Journal({ tripCtx, addToast }) {
  const [entries, setEntries] = useState([]);
  const [text, setText] = useState("");
  const [destination, setDestination] = useState(tripCtx?.destination || "");
  const [mood, setMood] = useState("happy");
  const [photoUrl, setPhotoUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState("all");

  const headers = () => {
    const h = { "Content-Type": "application/json" };
    const t = localStorage.getItem("sr_token");
    if (t) h.Authorization = `Bearer ${t}`;
    return h;
  };

  const loadEntries = async () => {
    try {
      const r = await fetch("/api/journal/entries", { headers: headers() });
      const d = await r.json();
      if (d.ok) {
        setEntries(d.entries || []);
        // Mirror to localStorage for offline reads
        localStorage.setItem("sr_journal", JSON.stringify(d.entries || []));
      }
    } catch {
      // Fallback to localStorage if API fails
      const cached = JSON.parse(localStorage.getItem("sr_journal") || "[]");
      setEntries(cached);
    }
  };

  useEffect(() => { loadEntries(); }, []);
  useEffect(() => { if (tripCtx?.destination) setDestination(tripCtx.destination); }, [tripCtx?.destination]);

  const save = async () => {
    if (!text.trim()) { addToast("Write something first", "warning"); return; }
    setLoading(true);
    try {
      const r = await fetch("/api/journal/entries", {
        method: "POST", headers: headers(),
        body: JSON.stringify({ text: text.trim(), destination: destination || "General", mood, photoUrl: photoUrl || null }),
      });
      const d = await r.json();
      if (d.ok) {
        addToast("Journal entry saved! 📝", "success");
        setText(""); setPhotoUrl("");
        await loadEntries();
      } else { addToast(d.error || "Failed to save", "error"); }
    } catch (e) { addToast(e.message, "error"); }
    finally { setLoading(false); }
  };

  const remove = async (id) => {
    try {
      await fetch(`/api/journal/entries?id=${id}`, { method: "DELETE", headers: headers() });
      addToast("Entry deleted", "success");
      await loadEntries();
    } catch (e) { addToast(e.message, "error"); }
  };

  const filtered = filter === "all" ? entries : entries.filter(e => e.mood === filter);

  return (
    <div style={{ padding: "20px 24px", maxWidth: 980, margin: "0 auto" }}>
      <motion.div initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }}
        style={{ marginBottom: 16 }}>
        <h1 style={{ fontSize: 26, fontWeight: 800, color: "var(--text)", marginBottom: 4 }}>📝 Trip Journal</h1>
        <p style={{ fontSize: 13, color: "var(--text-2)" }}>Capture moments, moods and memories from every place you visit. Stored on your account + offline.</p>
      </motion.div>

      <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }}
        className="card" style={{ padding: 20, marginBottom: 18 }}>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 10 }}>
          <input value={destination} onChange={e => setDestination(e.target.value)}
            placeholder="Destination (e.g. Goa, Jaipur…)"
            style={{ flex: "1 1 220px", padding: "9px 12px", borderRadius: 8, border: "1.5px solid var(--border)", background: "var(--bg)", color: "var(--text)", fontSize: 13 }} />
          <input value={photoUrl} onChange={e => setPhotoUrl(e.target.value)}
            placeholder="Photo URL (optional)"
            style={{ flex: "1 1 220px", padding: "9px 12px", borderRadius: 8, border: "1.5px solid var(--border)", background: "var(--bg)", color: "var(--text)", fontSize: 13 }} />
        </div>
        <textarea value={text} onChange={e => setText(e.target.value)} rows={4}
          placeholder="What did you experience today? Sights, food, feelings…"
          style={{ width: "100%", padding: "10px 12px", borderRadius: 8, border: "1.5px solid var(--border)", background: "var(--bg)", color: "var(--text)", fontSize: 13.5, lineHeight: 1.5, resize: "vertical", fontFamily: "inherit" }} />
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 10 }}>
          {MOODS.map(m => (
            <button key={m.id} onClick={() => setMood(m.id)}
              className={mood === m.id ? "chip active" : "chip"}
              style={mood === m.id ? { background: m.color + "33", borderColor: m.color, color: m.color } : {}}>
              {m.emoji} {m.label}
            </button>
          ))}
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 12, flexWrap: "wrap", gap: 8 }}>
          <span style={{ fontSize: 11.5, color: "var(--text-3)" }}>{text.length}/2000 chars</span>
          <button className="btn btn-primary" onClick={save} disabled={loading || !text.trim()}>
            {loading ? "Saving…" : "💾 Save Entry"}
          </button>
        </div>
      </motion.div>

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12, flexWrap: "wrap", gap: 8 }}>
        <h2 style={{ fontSize: 16, fontWeight: 700, color: "var(--text)" }}>
          Your entries · {entries.length}
        </h2>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          <button className={filter === "all" ? "chip active" : "chip"} onClick={() => setFilter("all")}>All</button>
          {MOODS.map(m => (
            <button key={m.id} className={filter === m.id ? "chip active" : "chip"} onClick={() => setFilter(m.id)}>
              {m.emoji}
            </button>
          ))}
        </div>
      </div>

      {filtered.length === 0 ? (
        <div className="card" style={{ padding: 32, textAlign: "center", color: "var(--text-3)", fontSize: 13 }}>
          {entries.length === 0 ? "No entries yet — write your first travel memory above! 📝" : "No entries match this mood."}
        </div>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(280px,1fr))", gap: 12 }}>
          <AnimatePresence>
            {filtered.map((e, i) => {
              const moodMeta = MOODS.find(m => m.id === e.mood) || MOODS[0];
              return (
                <motion.div key={e.id}
                  initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, scale: 0.95 }}
                  transition={{ delay: i * 0.04 }}
                  whileHover={{ y: -3, boxShadow: "var(--shadow-lg)" }}
                  className="card"
                  style={{ padding: 0, overflow: "hidden", borderLeft: `4px solid ${moodMeta.color}`, position: "relative" }}>
                  {e.photoUrl && (
                    <img src={e.photoUrl} alt={e.destination} loading="lazy"
                      style={{ width: "100%", height: 140, objectFit: "cover", display: "block" }} />
                  )}
                  <div style={{ padding: "12px 14px" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6, gap: 8 }}>
                      <span style={{ fontSize: 12, fontWeight: 700, color: moodMeta.color }}>
                        {moodMeta.emoji} {e.destination}
                      </span>
                      <span style={{ fontSize: 10.5, color: "var(--text-3)" }}>{e.dateLabel}</span>
                    </div>
                    <p style={{ fontSize: 13, color: "var(--text-2)", lineHeight: 1.5, whiteSpace: "pre-wrap", margin: 0 }}>{e.text}</p>
                    <button onClick={() => remove(e.id)}
                      style={{ position: "absolute", top: 6, right: 6, width: 22, height: 22, borderRadius: "50%", border: "none", background: "rgba(0,0,0,0.2)", color: "#fff", cursor: "pointer", fontSize: 11, display: "flex", alignItems: "center", justifyContent: "center" }}
                      title="Delete">×</button>
                  </div>
                </motion.div>
              );
            })}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
}
