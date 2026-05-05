import { useState } from "react";
import { Link } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";

export default function Login({ onAuth }) {
  const [form, setForm]       = useState({ email: "", password: "" });
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState("");

  const submit = async (e) => {
    e.preventDefault();
    if (!form.email || !form.password) { setError("Please fill all fields"); return; }
    setLoading(true);
    setError("");
    try {
      // Robust fetch — never throws on empty / non-JSON responses
      let result = { ok: false, data: null, error: "Login failed" };
      try {
        const res = await fetch("/api/auth/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email: form.email, password: form.password }),
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

      // Show a real backend error (e.g. wrong password) — but ONLY if the
      // server actually responded with a structured error.  Empty / network
      // / parse failures fall back to the client-side demo mode so the
      // hackathon demo still works without a running server.
      if (result.error && result.data && result.data.error) {
        setError(result.error);
      } else {
        const name = form.email.split("@")[0];
        const user = { name, initials: name.slice(0, 2).toUpperCase(), email: form.email };
        localStorage.setItem("sr_token", "demo-" + Date.now());
        onAuth(user);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-shell">
      {/* Left panel */}
      <div className="auth-left">
        <div className="auth-left-content">
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 48 }}>
            <div className="brand-logo" style={{ width: 46, height: 46, borderRadius: 12 }}>
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
              </svg>
            </div>
            <div>
              <div style={{ fontFamily: "'Sora',sans-serif", fontSize: 18, fontWeight: 800, color: "white", letterSpacing: "-0.02em" }}>SmartRoute</div>
              <div style={{ fontSize: 11, color: "rgba(255,255,255,0.55)", letterSpacing: "0.07em", textTransform: "uppercase", fontWeight: 600 }}>SRMIST Chapter</div>
            </div>
          </div>

          <h2 style={{ fontFamily: "'Sora',sans-serif", fontSize: "clamp(26px,3vw,42px)", fontWeight: 800, lineHeight: 1.1, letterSpacing: "-0.03em", color: "white", marginBottom: 16 }}>
            Plan smarter.<br/><span style={{ color: "#93c5fd" }}>Travel better.</span>
          </h2>
          <p style={{ fontSize: 14, color: "rgba(255,255,255,0.55)", maxWidth: 300, lineHeight: 1.7, marginBottom: 40 }}>
            Multi-agent AI travel planner — real-time data, smart budgets, and crowd intelligence for SRMIST students.
          </p>

          {["7 AI agents with Q-Learning & MCTS algorithms", "Live map with crowd density predictions", "Smart budget optimizer per travel persona", "Language tips for every Indian destination"].map(f => (
            <div key={f} style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 13, color: "rgba(255,255,255,0.75)", marginBottom: 12 }}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.5)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="20 6 9 17 4 12"/>
              </svg>
              {f}
            </div>
          ))}
        </div>
      </div>

      {/* Right form */}
      <div className="auth-right">
        <motion.div className="auth-form-box" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
          <div className="auth-title">Welcome back</div>
          <div className="auth-sub">Sign in to your SmartRoute account</div>

          <form className="auth-form" onSubmit={submit}>
            <div className="field-group">
              <label className="field-label">Email address</label>
              <input type="email" className="field-input" placeholder="you@srmist.edu.in"
                value={form.email} onChange={e => setForm(f => ({ ...f, email: e.target.value }))} autoComplete="email"/>
            </div>
            <div className="field-group">
              <label className="field-label">Password</label>
              <input type="password" className="field-input" placeholder="••••••••"
                value={form.password} onChange={e => setForm(f => ({ ...f, password: e.target.value }))} autoComplete="current-password"/>
            </div>

            <AnimatePresence>
              {error && (
                <motion.div initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                  style={{ fontSize: 13, color: "var(--red)", padding: "10px 14px", background: "var(--red-bg)", borderRadius: "var(--r-md)", border: "1.5px solid rgba(220,38,38,0.2)" }}>
                  {error}
                </motion.div>
              )}
            </AnimatePresence>

            <button type="submit" className="btn btn-primary w-full" style={{ height: 46 }} disabled={loading}>
              {loading ? "Signing in..." : "Sign in →"}
            </button>

            <div className="auth-divider">or</div>

            <button type="button" className="btn btn-ghost w-full" style={{ height: 46 }}
              onClick={() => onAuth({ name: "Demo User", initials: "DU", email: "demo@srmist.edu.in" })}>
              Continue as Demo User
            </button>
          </form>

          <p style={{ marginTop: 20, textAlign: "center", fontSize: 13, color: "var(--text-3)" }}>
            Don't have an account? <Link to="/register" className="auth-link">Create one →</Link>
          </p>
        </motion.div>
      </div>
    </div>
  );
}
