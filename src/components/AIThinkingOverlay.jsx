import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useState } from "react";

const thinkingSteps = [
  { agent: "Preference Agent", text: "Analyzing your travel persona and interests...", icon: "❤️" },
  { agent: "Weather Agent", text: "Fetching real-time weather forecasts...", icon: "🌦️" },
  { agent: "Crowd Agent", text: "Predicting crowd density patterns...", icon: "👥" },
  { agent: "Budget Agent", text: "Optimizing budget allocation with MDP...", icon: "💰" },
  { agent: "Planner Agent", text: "Running Monte Carlo Tree Search for routes...", icon: "🛸" },
  { agent: "Booking Agent", text: "Scanning flights, hotels, and activities...", icon: "🎫" },
  { agent: "Explainability Agent", text: "Generating decision reasoning trail...", icon: "🧩" }
];

const pipelineStages = ["Observe", "Infer", "Decide", "Plan", "Act"];

function AIThinkingOverlay({ visible, onDismiss }) {
  const [currentStep, setCurrentStep] = useState(0);
  const [currentStage, setCurrentStage] = useState(0);

  useEffect(() => {
    if (!visible) {
      setCurrentStep(0);
      setCurrentStage(0);
      return;
    }

    const stepInterval = setInterval(() => {
      setCurrentStep(prev => (prev + 1) % thinkingSteps.length);
    }, 1800);

    const stageInterval = setInterval(() => {
      setCurrentStage(prev => Math.min(prev + 1, pipelineStages.length - 1));
    }, 2200);

    return () => {
      clearInterval(stepInterval);
      clearInterval(stageInterval);
    };
  }, [visible]);

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          className="ai-thinking-overlay"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.4 }}
          onClick={onDismiss}
        >
          {/* Neural network background */}
          <div className="thinking-neural-bg">
            {Array.from({ length: 12 }).map((_, i) => (
              <motion.div
                key={i}
                className="neural-node"
                style={{
                  left: `${10 + (i % 4) * 25}%`,
                  top: `${15 + Math.floor(i / 4) * 30}%`
                }}
                animate={{
                  scale: [1, 1.3, 1],
                  opacity: [0.15, 0.4, 0.15]
                }}
                transition={{
                  duration: 2 + (i % 3) * 0.5,
                  repeat: Infinity,
                  delay: i * 0.2
                }}
              />
            ))}
            {Array.from({ length: 8 }).map((_, i) => (
              <motion.div
                key={`line-${i}`}
                className="neural-line"
                style={{
                  left: `${15 + (i % 3) * 30}%`,
                  top: `${20 + Math.floor(i / 3) * 25}%`,
                  width: `${60 + (i % 4) * 20}px`,
                  transform: `rotate(${-30 + i * 18}deg)`
                }}
                animate={{ opacity: [0.05, 0.2, 0.05] }}
                transition={{
                  duration: 1.5 + (i % 2),
                  repeat: Infinity,
                  delay: i * 0.15
                }}
              />
            ))}
          </div>

          <motion.div
            className="thinking-content"
            initial={{ scale: 0.92, y: 20 }}
            animate={{ scale: 1, y: 0 }}
            exit={{ scale: 0.95, y: 10 }}
            transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
            onClick={e => e.stopPropagation()}
          >
            {/* Animated brain icon */}
            <motion.div
              className="thinking-brain"
              animate={{ rotate: [0, 5, -5, 0] }}
              transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
            >
              <div className="thinking-brain-ring" />
              <span>🧠</span>
            </motion.div>

            <h3 className="thinking-title">AI Pipeline Active</h3>
            <p className="thinking-subtitle">7 agents collaborating in real-time</p>

            {/* Pipeline stages flow */}
            <div className="thinking-stages">
              {pipelineStages.map((stage, i) => (
                <div key={stage} className="thinking-stage-wrapper">
                  <motion.div
                    className={`thinking-stage ${i <= currentStage ? "active" : ""} ${i === currentStage ? "current" : ""}`}
                    animate={i === currentStage ? { scale: [1, 1.08, 1] } : {}}
                    transition={{ duration: 1.2, repeat: Infinity }}
                  >
                    <span className="thinking-stage-dot" />
                    <span className="thinking-stage-label">{stage}</span>
                  </motion.div>
                  {i < pipelineStages.length - 1 && (
                    <motion.div
                      className={`thinking-stage-connector ${i < currentStage ? "active" : ""}`}
                      animate={i === currentStage ? { opacity: [0.3, 1, 0.3] } : {}}
                      transition={{ duration: 0.8, repeat: Infinity }}
                    />
                  )}
                </div>
              ))}
            </div>

            {/* Current agent activity */}
            <AnimatePresence mode="wait">
              <motion.div
                key={currentStep}
                className="thinking-agent-card"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.3 }}
              >
                <span className="thinking-agent-icon">{thinkingSteps[currentStep].icon}</span>
                <div className="thinking-agent-info">
                  <strong>{thinkingSteps[currentStep].agent}</strong>
                  <p>{thinkingSteps[currentStep].text}</p>
                </div>
                <motion.span
                  className="thinking-agent-pulse"
                  animate={{ scale: [1, 1.4, 1], opacity: [1, 0.4, 1] }}
                  transition={{ duration: 1, repeat: Infinity }}
                />
              </motion.div>
            </AnimatePresence>

            {/* Progress bar */}
            <div className="thinking-progress-track">
              <motion.div
                className="thinking-progress-fill"
                animate={{ width: ["0%", "100%"] }}
                transition={{ duration: 12, ease: "linear" }}
              />
            </div>

            <p className="thinking-dismiss-hint">Click anywhere to dismiss</p>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

export default AIThinkingOverlay;
