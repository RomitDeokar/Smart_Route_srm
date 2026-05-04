import { useState } from "react";
import { Link } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";

export default function Register({ onAuth }) {
  const [form, setForm]       = useState({ name: "", email: "", password: "", confirm: "" });
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState("");

  const submit = async (e) => {
    e.preventDefault();
    if (!form.name || !form.email || !form.password) { setError("Please fill all fields"); return; }
    if (form.password !== form.confirm) { setError("Passwords don't match"); return; }
    if (form.password.length < 6) { setError("Password must be at least 6 characters"); return; }
    setLoading(true); setError("");
    try {
      // Robust fetch — never throws on empty / non-JSON responses
      let result = { ok: false, data: null, error: "Registration failed" };
      try {
        const res = await fetch("/api/auth/register", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: form.name, email: form.email, password: form.password }),
        });
        const raw = await res.text();
        let parsed = null;
        try { parsed = raw ? JSON.parse(raw) : null; } catch { parsed = null; }
        result = { ok: res.ok && parsed?.ok, data: parsed,
                   error: parsed?.error || (!res.ok ? `HTTP ${res.status}` : null) };
      } catch (netErr) {
        result = { ok: false, data: null, error: netErr?.message || "Network error" };
      }

      if (result.ok && result.data?.token) {
        localStorage.setItem("sr_token", result.data.token);
        onAuth(result.data.user);
        return;
      }

      // Real backend error → show it.  Network / empty body → demo fallback.
      if (result.error && result.data && result.data.error) {
        setError(result.error);
      } else {
        const initials = form.name.split(" ").map(w => w[0]).join("").slice(0, 2).toUpperCase();
        const user = { name: form.name, initials, email: form.email };
        localStorage.setItem("sr_token", "demo-" + Date.now());
        onAuth(user);
      }
    } finally { setLoading(false); }
  };

  return (
    <div className="auth-shell">
      <div className="auth-left">
        <div className="auth-left-content">
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 48 }}>
            <div className="brand-logo" style={{ width: 46, height: 46, borderRadius: 12 }}>
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
              </svg>
            </div>
            <div>
              <div style={{ fontFamily: "'Sora',sans-serif", fontSize: 18, fontWeight: 800, color: "white" }}>SmartRoute</div>
              <div style={{ fontSize: 11, color: "rgba(255,255,255,0.55)", letterSpacing: "0.07em", textTransform: "uppercase", fontWeight: 600 }}>SRMIST Chapter</div>
            </div>
          </div>
          <h2 style={{ fontFamily: "'Sora',sans-serif", fontSize: "clamp(24px,3vw,38px)", fontWeight: 800, lineHeight: 1.1, color: "white", marginBottom: 16, letterSpacing: "-0.02em" }}>
            Your AI travel<br/><span style={{ color: "#93c5fd" }}>companion awaits.</span>
          </h2>
          <p style={{ fontSize: 14, color: "rgba(255,255,255,0.55)", maxWidth: 300, lineHeight: 1.7, marginBottom: 40 }}>
            Join SRMIST students planning smarter trips with real AI.
          </p>
          {["Multi-agent AI pipeline (7 agents)", "Real-time crowd and weather analysis", "Language tips for every destination", "AI-generated trip checklists"].map(f => (
            <div key={f} style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 13, color: "rgba(255,255,255,0.75)", marginBottom: 12 }}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.5)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="20 6 9 17 4 12"/>
              </svg>
              {f}
            </div>
          ))}
        </div>
      </div>

      <div className="auth-right">
        <motion.div className="auth-form-box" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
          <div className="auth-title">Create account</div>
          <div className="auth-sub">Start planning smarter trips today</div>
          <form className="auth-form" onSubmit={submit}>
            <div className="field-group">
              <label className="field-label">Full name</label>
              <input className="field-input" placeholder="Your full name" value={form.name}
                onChange={e => setForm(f => ({ ...f, name: e.target.value }))} />
            </div>
            <div className="field-group">
              <label className="field-label">Email address</label>
              <input type="email" className="field-input" placeholder="you@srmist.edu.in" value={form.email}
                onChange={e => setForm(f => ({ ...f, email: e.target.value }))} />
            </div>
            <div className="grid-2">
              <div className="field-group">
                <label className="field-label">Password</label>
                <input type="password" className="field-input" placeholder="Min 6 chars" value={form.password}
                  onChange={e => setForm(f => ({ ...f, password: e.target.value }))} />
              </div>
              <div className="field-group">
                <label className="field-label">Confirm</label>
                <input type="password" className="field-input" placeholder="Repeat" value={form.confirm}
                  onChange={e => setForm(f => ({ ...f, confirm: e.target.value }))} />
              </div>
            </div>
            <AnimatePresence>
              {error && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  style={{ fontSize: 13, color: "var(--red)", padding: "10px 14px", background: "var(--red-bg)", borderRadius: "var(--r-md)", border: "1.5px solid rgba(220,38,38,0.2)" }}>
                  {error}
                </motion.div>
              )}
            </AnimatePresence>
            <button type="submit" className="btn btn-primary w-full" style={{ height: 46 }} disabled={loading}>
              {loading ? "Creating account..." : "Create account →"}
            </button>
          </form>
          <p style={{ marginTop: 20, textAlign: "center", fontSize: 13, color: "var(--text-3)" }}>
            Already have an account? <Link to="/login" className="auth-link">Sign in →</Link>
          </p>
        </motion.div>
      </div>
    </div>
  );
}
