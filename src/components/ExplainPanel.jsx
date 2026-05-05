/* ExplainPanel.jsx — "Why this plan?" explainability UI */
import { motion, AnimatePresence } from "framer-motion";
import { useState } from "react";

const FACTOR_ICONS = {
  weatherSafety:      "🌤️",
  budgetCompliance:   "💰",
  preferenceMatch:    "❤️",
  crowdAvoidance:     "👥",
  routeEfficiency:    "🛸",
  bookingFeasibility: "🎫",
  emergencyBuffer:    "🛡️",
};

const IMPACT_COLORS = {
  high_positive: "var(--green)",
  positive:      "var(--accent-light)",
  neutral:       "var(--text-2)",
  risk:          "var(--red)",
};

function ConfidenceMeter({ score }) {
  const pct   = Math.round((score || 0) * 100);
  const color = pct >= 85 ? "var(--green)" : pct >= 70 ? "var(--gold)" : "var(--red)";
  const label = pct >= 85 ? "High" : pct >= 70 ? "Moderate" : "Low";

  return (
    <div style={{ textAlign: "center", padding: "16px 0 8px" }}>
      <svg width="120" height="120" viewBox="0 0 120 120">
        {/* Track */}
        <circle cx="60" cy="60" r="48" fill="none" stroke="var(--bg-raised)" strokeWidth="10" />
        {/* Fill */}
        <circle
          cx="60" cy="60" r="48"
          fill="none" stroke={color} strokeWidth="10"
          strokeDasharray={`${2 * Math.PI * 48 * pct / 100} ${2 * Math.PI * 48}`}
          strokeLinecap="round"
          transform="rotate(-90 60 60)"
          style={{ transition: "stroke-dasharray 0.8s ease" }}
        />
        <text x="60" y="54" textAnchor="middle" fill="var(--text)" fontSize="22" fontWeight="800" fontFamily="Outfit,sans-serif">
          {pct}%
        </text>
        <text x="60" y="72" textAnchor="middle" fill="var(--text-2)" fontSize="11" fontFamily="Outfit,sans-serif">
          confidence
        </text>
      </svg>
      <div style={{ fontSize: 13, fontWeight: 600, color, marginTop: 4 }}>{label} Confidence</div>
    </div>
  );
}

function FactorBar({ label, icon, value, maxVal = 0.25 }) {
  const pct = Math.min(100, Math.round((value / maxVal) * 100));
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, fontSize: 12.5 }}>
        <span style={{ color: "var(--text-2)", display: "flex", alignItems: "center", gap: 5 }}>
          <span style={{ fontSize: 14 }}>{icon}</span>{label}
        </span>
        <span style={{ color: "var(--text)", fontWeight: 600 }}>{(value * 100).toFixed(1)}pts</span>
      </div>
      <div className="progress-wrap">
        <motion.div
          className="progress-fill"
          style={{ background: "var(--g-accent)" }}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.6, delay: 0.1 }}
        />
      </div>
    </div>
  );
}

export default function ExplainPanel({ pipeline, summary, isOpen, onClose }) {
  const [tab, setTab] = useState("confidence");

  const decision = pipeline?.decision || {};
  const agents   = pipeline?.agents   || [];
  const stages   = pipeline?.stages   || [];
  const planning = pipeline?.planning  || {};

  const confScore    = decision.confidenceScore   || summary?.confidence || 0.88;
  const attribution  = decision.factorAttribution || {};
  const reasoning    = decision.decisionReasoning || [];
  const whyThisPlan  = decision.whyThisPlan       || [];
  const sensitivity  = decision.sensitivityAnalysis || [];
  const interval     = decision.confidenceInterval  || {};

  const TABS = [
    { id: "confidence", label: "Confidence" },
    { id: "factors",    label: "Factors" },
    { id: "reasoning",  label: "Reasoning" },
    { id: "pipeline",   label: "Pipeline" },
  ];

  if (!isOpen) return null;

  return (
    <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.65)", backdropFilter: "blur(8px)", zIndex: 1000, display: "flex", alignItems: "center", justifyContent: "center", padding: 16 }}>
      <motion.div
        initial={{ scale: 0.92, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.95, opacity: 0 }}
        transition={{ duration: 0.22 }}
        style={{ background: "white", border: "1px solid var(--border-mid)", borderRadius: "var(--r-2xl)", width: "100%", maxWidth: 560, maxHeight: "88vh", overflow: "hidden", display: "flex", flexDirection: "column" }}
      >
        {/* Header */}
        <div style={{ padding: "16px 20px 12px", borderBottom: "1px solid var(--border)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div>
            <div style={{ fontSize: 15, fontWeight: 700, color: "var(--text)" }}>✦ Explainability Layer</div>
            <div style={{ fontSize: 12, color: "var(--text-2)", marginTop: 2 }}>Why this plan was generated</div>
          </div>
          <button onClick={onClose} style={{ background: "none", border: "none", cursor: "pointer", color: "var(--text-2)", fontSize: 20, lineHeight: 1 }}>×</button>
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 4, padding: "10px 16px 0", borderBottom: "1px solid var(--border)" }}>
          {TABS.map(t => (
            <button key={t.id} onClick={() => setTab(t.id)}
              style={{ padding: "5px 12px", borderRadius: "var(--r-sm) var(--r-sm) 0 0", border: "none", cursor: "pointer", fontSize: 12.5, fontWeight: 600,
                background: tab === t.id ? "var(--bg-raised)" : "transparent",
                color: tab === t.id ? "var(--text)" : "var(--text-2)",
                borderBottom: tab === t.id ? "2px solid var(--accent)" : "2px solid transparent",
              }}>
              {t.label}
            </button>
          ))}
        </div>

        {/* Tab content */}
        <div style={{ flex: 1, overflowY: "auto", padding: "16px 20px" }}>

          {tab === "confidence" && (
            <div>
              <ConfidenceMeter score={confScore} />

              {interval.lower && (
                <div style={{ background: "var(--bg-soft)", border: "1px solid var(--border)", borderRadius: "var(--r-md)", padding: "10px 14px", marginTop: 8, fontSize: 12.5 }}>
                  <div style={{ color: "var(--text-2)", marginBottom: 4 }}>Confidence interval (95%)</div>
                  <div style={{ display: "flex", justifyContent: "space-between", color: "var(--text)" }}>
                    <span>Lower: <strong>{Math.round(interval.lower * 100)}%</strong></span>
                    <span>Point: <strong>{Math.round(confScore * 100)}%</strong></span>
                    <span>Upper: <strong>{Math.round(interval.upper * 100)}%</strong></span>
                  </div>
                </div>
              )}

              {whyThisPlan.length > 0 && (
                <div style={{ marginTop: 14 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text-2)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>Why this plan?</div>
                  {whyThisPlan.map((line, i) => (
                    <div key={i} style={{ display: "flex", gap: 8, marginBottom: 7, fontSize: 13, color: "var(--text-2)", lineHeight: 1.5 }}>
                      <span style={{ color: "var(--accent)", flexShrink: 0, marginTop: 2 }}>→</span>
                      {line}
                    </div>
                  ))}
                </div>
              )}

              {/* Agent scores row */}
              {agents.length > 0 && (
                <div style={{ marginTop: 14, display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8 }}>
                  {agents.slice(0, 6).map(ag => (
                    <div key={ag.agent} style={{ background: "var(--bg-soft)", border: "1px solid var(--border)", borderRadius: "var(--r-md)", padding: "8px 10px", textAlign: "center" }}>
                      <div style={{ fontSize: 16, fontWeight: 800, color: ag.score >= 0.85 ? "var(--green)" : ag.score >= 0.70 ? "var(--gold)" : "var(--red)" }}>
                        {Math.round((ag.score || 0) * 100)}%
                      </div>
                      <div style={{ fontSize: 10.5, color: "var(--text-2)", marginTop: 2, lineHeight: 1.3 }}>
                        {ag.agent.split(" (")[0].replace("Agent", "").trim()}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {tab === "factors" && (
            <div>
              <div style={{ fontSize: 12, color: "var(--text-2)", marginBottom: 14, lineHeight: 1.5 }}>
                SHAP-style attribution — how much each factor contributed to the confidence score.
              </div>
              {Object.entries(attribution).map(([key, val]) => (
                <FactorBar
                  key={key}
                  label={key.replace(/([A-Z])/g, " $1").replace(/^./, s => s.toUpperCase())}
                  icon={FACTOR_ICONS[key] || "◉"}
                  value={val}
                  maxVal={0.22}
                />
              ))}

              {sensitivity.length > 0 && (
                <div style={{ marginTop: 16 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text-2)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>Sensitivity ranking</div>
                  {sensitivity.map((f, i) => (
                    <div key={f.factor} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 7, padding: "7px 10px", background: "var(--bg-soft)", borderRadius: "var(--r-sm)", border: "1px solid var(--border)" }}>
                      <span style={{ fontSize: 13, fontWeight: 700, color: "var(--text-2)", minWidth: 18 }}>#{i + 1}</span>
                      <span style={{ flex: 1, fontSize: 13, color: "var(--text)" }}>{f.humanLabel}</span>
                      <span style={{ fontSize: 12, fontWeight: 600, color: IMPACT_COLORS[f.impact] || "var(--text-2)" }}>
                        {(f.score * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {tab === "reasoning" && (
            <div>
              {reasoning.map((line, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -8 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.06 }}
                  style={{ display: "flex", gap: 10, marginBottom: 12, padding: "10px 12px", background: "var(--bg-soft)", borderRadius: "var(--r-md)", border: "1px solid var(--border)" }}
                >
                  <span style={{ width: 22, height: 22, borderRadius: "50%", background: "var(--accent-dim)", border: "1px solid rgba(59,130,246,0.2)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 700, color: "var(--accent-light)", flexShrink: 0 }}>{i + 1}</span>
                  <span style={{ fontSize: 13, color: "var(--text-2)", lineHeight: 1.55 }}>{line}</span>
                </motion.div>
              ))}
            </div>
          )}

          {tab === "pipeline" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {/* MCTS info */}
              {planning.mcts && (
                <div style={{ background: "var(--bg-soft)", border: "1px solid var(--border)", borderRadius: "var(--r-md)", padding: "12px 14px" }}>
                  <div style={{ fontSize: 12.5, fontWeight: 600, color: "var(--accent-light)", marginBottom: 6 }}>MCTS Route Planner</div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, fontSize: 12 }}>
                    {[
                      ["Iterations", planning.mcts.iterations],
                      ["Nodes explored", planning.mcts.nodesExplored],
                      ["Best path score", `${(planning.mcts.bestPathScore * 100).toFixed(1)}%`],
                      ["Convergence", `${(planning.mcts.convergenceRate * 100).toFixed(0)}%`],
                    ].map(([k, v]) => (
                      <div key={k} style={{ display: "flex", justifyContent: "space-between" }}>
                        <span style={{ color: "var(--text-2)" }}>{k}</span>
                        <span style={{ color: "var(--text)", fontWeight: 600 }}>{v}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Pipeline stages */}
              {stages.map((stage, i) => (
                <div key={stage.id} style={{ display: "flex", alignItems: "center", gap: 10, padding: "9px 12px", background: "var(--bg-soft)", border: "1px solid var(--border)", borderRadius: "var(--r-md)" }}>
                  <div style={{ width: 22, height: 22, borderRadius: "50%", background: "var(--green-d)", border: "1px solid rgba(34,197,94,0.2)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, color: "var(--green)", flexShrink: 0 }}>✓</div>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text)" }}>{stage.name}</div>
                    <div style={{ fontSize: 11.5, color: "var(--text-2)" }}>{stage.detail}</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </motion.div>
    </div>
  );
}
