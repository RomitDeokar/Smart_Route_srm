import { motion } from "framer-motion";
import { useState } from "react";

const pipelineNodes = [
  {
    id: "observe",
    name: "Observe",
    icon: "📡",
    iconClass: "capture",
    desc: "Capture user intent, validate parameters, prepare request payload.",
    stage: "Input"
  },
  {
    id: "infer",
    name: "Infer",
    icon: "🧠",
    iconClass: "analysis",
    desc: "Multi-agent analysis — preference, budget, weather, crowd agents collaborate.",
    stage: "Analysis"
  },
  {
    id: "decide",
    name: "Decide",
    icon: "⚖️",
    iconClass: "decision",
    desc: "MDP reward function + RL policy refinement with confidence scoring.",
    stage: "Policy"
  },
  {
    id: "plan",
    name: "Plan",
    icon: "🎯",
    iconClass: "planning",
    desc: "MCTS route optimization (50 iterations) + nearest-neighbor TSP.",
    stage: "Optimization"
  },
  {
    id: "act",
    name: "Act",
    icon: "🚀",
    iconClass: "llm",
    desc: "Generate itinerary, book options, explain decisions, deliver results.",
    stage: "Execution"
  }
];

const agentMapping = {
  observe: ["Preference Agent"],
  infer: ["Weather Agent", "Crowd Agent", "Budget Agent"],
  decide: ["Explainability Agent"],
  plan: ["Planner Agent"],
  act: ["Booking Agent"]
};

const staggerItem = {
  hidden: { opacity: 0, y: 16 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.4, ease: "easeOut" } }
};

function PipelineVisualization({ pipelineData, loading }) {
  const stages = pipelineData?.stages || [];
  const [expandedNode, setExpandedNode] = useState(null);

  const getNodeStatus = (nodeId) => {
    if (loading) {
      const stageIndex = pipelineNodes.findIndex(n => n.id === nodeId);
      const completedCount = stages.filter(s => s.status === "completed").length;
      if (stageIndex < completedCount) return "completed";
      if (stageIndex === completedCount) return "running";
      return "idle";
    }
    const stage = stages.find(s => s.id === nodeId);
    return stage?.status || (stages.length > 0 ? "completed" : "idle");
  };

  const getNodeDetail = (nodeId) => {
    const stage = stages.find(s => s.id === nodeId);
    return stage?.detail || null;
  };

  const toggleExpand = (nodeId) => {
    setExpandedNode(prev => prev === nodeId ? null : nodeId);
  };

  return (
    <motion.section
      className="glass-panel pipeline-section pipeline-v2"
      id="pipeline-section"
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.1 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
    >
      <div className="section-label">⚙️ Architecture</div>
      <h2 className="section-title">Agentic AI Pipeline</h2>
      <p className="section-subtitle">
        From observation to action — watch the multi-agent pipeline process your request in real-time.
      </p>

      {/* ── Horizontal Animated Flow ── */}
      <div className="pipeline-flow-v2">
        {pipelineNodes.map((node, i) => {
          const status = getNodeStatus(node.id);
          const detail = getNodeDetail(node.id);
          const isExpanded = expandedNode === node.id;
          const agents = agentMapping[node.id] || [];

          return (
            <div key={node.id} className="pipeline-flow-item">
              <motion.div
                className={`pipeline-node-v2 ${status}`}
                variants={staggerItem}
                whileHover={{ y: -6, scale: 1.03 }}
                onClick={() => toggleExpand(node.id)}
                style={{ cursor: "pointer" }}
              >
                {/* Top glow bar */}
                <motion.div
                  className="pipeline-node-glow"
                  animate={status === "running" ? {
                    opacity: [0.4, 1, 0.4],
                    boxShadow: ["0 0 8px currentColor", "0 0 20px currentColor", "0 0 8px currentColor"]
                  } : {}}
                  transition={{ duration: 1.5, repeat: Infinity }}
                />

                {/* Stage label */}
                <div className="pipeline-stage-label">{node.stage}</div>

                {/* Icon */}
                <motion.div
                  className={`pipeline-icon-v2 ${node.iconClass}`}
                  animate={status === "running" ? { scale: [1, 1.12, 1] } : {}}
                  transition={{ duration: 1.2, repeat: Infinity }}
                >
                  {node.icon}
                </motion.div>

                {/* Name */}
                <div className="pipeline-name-v2">{node.name}</div>

                {/* Status Badge */}
                <div className={`pipeline-status-v2 ${status}`}>
                  <motion.span
                    className="pipeline-status-dot-v2"
                    animate={status === "running" ? { scale: [1, 1.5, 1], opacity: [1, 0.5, 1] } : {}}
                    transition={{ duration: 1, repeat: Infinity }}
                  />
                  {status === "idle" ? "Standby" : status === "running" ? "Processing" : "Complete"}
                </div>

                {/* Agents involved */}
                <div className="pipeline-agents-mini">
                  {agents.map(a => (
                    <span key={a} className="pipeline-agent-chip">{a.split(" ")[0]}</span>
                  ))}
                </div>

                {/* Detail */}
                {detail && <div className="pipeline-detail-v2">{detail}</div>}
              </motion.div>

              {/* Connector Arrow */}
              {i < pipelineNodes.length - 1 && (
                <div className={`pipeline-connector-v2 ${getNodeStatus(pipelineNodes[i + 1].id) !== "idle" ? "active" : ""}`}>
                  <motion.div
                    className="pipeline-connector-particle"
                    animate={status === "completed" || status === "running" ? {
                      x: [0, 40],
                      opacity: [1, 0]
                    } : {}}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  />
                  <svg width="40" height="20" viewBox="0 0 40 20">
                    <path d="M0 10 L30 10 L25 5 M30 10 L25 15" fill="none" stroke="currentColor" strokeWidth="1.5" />
                  </svg>
                </div>
              )}

              {/* Why This Decision — expandable card */}
              {isExpanded && (node.id === "decide" || node.id === "plan") && (
                <motion.div
                  className="pipeline-why-card"
                  initial={{ opacity: 0, y: -10, height: 0 }}
                  animate={{ opacity: 1, y: 0, height: "auto" }}
                  exit={{ opacity: 0, y: -10, height: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <h5>💡 Why this decision?</h5>
                  <p className="pipeline-why-text">
                    {node.id === "decide"
                      ? "The MDP reward function weighted satisfaction (40%), budget efficiency (30%), weather suitability (20%), and crowd avoidance (10%) to select the optimal policy."
                      : "MCTS explored 50 route permutations with UCB1 selection. The winning route minimizes travel time while maximizing attraction ratings and budget compliance."}
                  </p>
                </motion.div>
              )}
            </div>
          );
        })}
      </div>

      {/* ── Agent Confidence Scores ── */}
      {pipelineData?.agents?.length > 0 && (
        <motion.div
          className="pipeline-agents-section"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          <h3 className="pipeline-agents-title">Agent Confidence Scores</h3>
          <div className="pipeline-agents-grid">
            {pipelineData.agents.map(agent => (
              <motion.div
                key={agent.agent}
                className="pipeline-agent-card"
                whileHover={{ y: -4, scale: 1.02 }}
              >
                <div className="pipeline-agent-score-ring">
                  <svg viewBox="0 0 72 72">
                    <circle cx="36" cy="36" r="30" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="5" />
                    <motion.circle
                      cx="36" cy="36" r="30"
                      fill="none"
                      stroke={agent.score > 0.8 ? "var(--green)" : "var(--gold)"}
                      strokeWidth="5"
                      strokeDasharray={`${agent.score * 188.5} 188.5`}
                      strokeLinecap="round"
                      initial={{ strokeDasharray: "0 188.5" }}
                      animate={{ strokeDasharray: `${agent.score * 188.5} 188.5` }}
                      transition={{ duration: 1, delay: 0.3, ease: "easeOut" }}
                      transform="rotate(-90 36 36)"
                    />
                  </svg>
                  <span className="pipeline-agent-score-value">
                    {(agent.score * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="pipeline-agent-card-name">
                  {agent.agent.split(" (")[0]}
                </div>
                <div className="pipeline-agent-card-detail">
                  {agent.output?.recommendation?.slice(0, 50) || `Confidence: ${(agent.score * 100).toFixed(0)}%`}
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}

      {/* ── Decision Output ── */}
      {pipelineData?.decision && (
        <motion.div
          className="pipeline-decision-block"
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
        >
          <div className="pipeline-decision-header">
            <h3>Decision Output</h3>
            <span className="pipeline-decision-score">
              {(pipelineData.decision.confidenceScore * 100).toFixed(1)}%
            </span>
          </div>

          <div className="pipeline-factors-grid">
            {Object.entries(pipelineData.decision.factorAttribution || {}).map(([key, value]) => (
              <div key={key} className="pipeline-factor-card">
                <div className="pipeline-factor-label">
                  {key.replace(/([A-Z])/g, " $1").trim()}
                </div>
                <div className="pipeline-factor-bar">
                  <motion.div
                    className="pipeline-factor-fill"
                    style={{ background: Number(value) > 0.8 ? "var(--green)" : "var(--gold)" }}
                    initial={{ width: 0 }}
                    animate={{ width: `${Number(value) * 100}%` }}
                    transition={{ duration: 0.8, delay: 0.2 }}
                  />
                </div>
                <div className="pipeline-factor-value" style={{ color: Number(value) > 0.8 ? "var(--green)" : "var(--gold)" }}>
                  {(Number(value) * 100).toFixed(0)}%
                </div>
              </div>
            ))}
          </div>

          <div className="pipeline-reasoning-list">
            {(pipelineData.decision.decisionReasoning || []).map((reason, i) => (
              <motion.p
                key={i}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.8 + i * 0.1 }}
              >
                {reason}
              </motion.p>
            ))}
          </div>
        </motion.div>
      )}
    </motion.section>
  );
}

export default PipelineVisualization;
