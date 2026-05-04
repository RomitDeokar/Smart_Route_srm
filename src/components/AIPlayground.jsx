import { motion } from "framer-motion";
import { useState, useMemo } from "react";
import FeatureShell from "./FeatureShell.jsx";

const sliderConfig = [
  { id: "budget", label: "Budget Level", min: 1, max: 10, icon: "💰", color: "var(--green)" },
  { id: "crowd", label: "Crowd Tolerance", min: 1, max: 10, icon: "👥", color: "var(--blue)" },
  { id: "weather", label: "Weather Sensitivity", min: 1, max: 10, icon: "🌦️", color: "var(--gold)" },
  { id: "adventure", label: "Adventure Level", min: 1, max: 10, icon: "⚡", color: "var(--violet)" }
];

function computePrediction(values) {
  const budget = values.budget || 5;
  const crowd = values.crowd || 5;
  const weather = values.weather || 5;
  const adventure = values.adventure || 5;

  const comfortScore = Math.round(((11 - crowd) * 0.35 + (11 - weather) * 0.25 + budget * 0.25 + (11 - adventure) * 0.15) * 10);
  const adventureRating = Math.round((adventure * 0.45 + (11 - budget) * 0.15 + crowd * 0.2 + weather * 0.2) * 10);
  const costEstimate = Math.round(3000 + budget * 2200 + adventure * 800);
  const optimalTime = crowd <= 4 ? "Early Morning" : crowd <= 7 ? "Late Afternoon" : "Off-Peak Hours";

  const persona = adventure >= 7 ? "Adventure Seeker" :
    budget <= 3 ? "Budget Hacker" :
    budget >= 8 ? "Luxury Traveler" : "Explorer";

  const tips = [];
  if (budget >= 8 && crowd <= 3) tips.push("Perfect combo for exclusive experiences");
  if (adventure >= 8 && weather >= 7) tips.push("Consider indoor alternatives for risky weather");
  if (crowd >= 8) tips.push("Book skip-the-line passes for popular spots");
  if (budget <= 3 && adventure >= 6) tips.push("Street food + free attractions = epic combo");
  if (tips.length === 0) tips.push("Well-balanced preferences for an enjoyable trip");

  return { comfortScore, adventureRating, costEstimate, optimalTime, persona, tips };
}

function AIPlayground({ onApply }) {
  const [values, setValues] = useState({
    budget: 5,
    crowd: 5,
    weather: 5,
    adventure: 5
  });

  const prediction = useMemo(() => computePrediction(values), [values]);

  const handleSliderChange = (id, val) => {
    setValues(prev => ({ ...prev, [id]: Number(val) }));
  };

  return (
    <FeatureShell
      feature="AI Lab"
      title="AI Playground"
      subtitle="Adjust parameters and see instant predictions"
      icon="🎮"
      defaultExpanded={false}
      action={
        onApply ? (
          <button className="button button-primary" onClick={() => onApply(values)}>
            Apply to Trip
          </button>
        ) : null
      }
    >
      <div className="playground-panel">
        {/* Sliders */}
        <div className="playground-sliders">
          {sliderConfig.map(slider => (
            <div key={slider.id} className="playground-slider-row">
              <div className="playground-slider-head">
                <span className="playground-slider-icon">{slider.icon}</span>
                <span className="playground-slider-label">{slider.label}</span>
                <span className="playground-slider-value" style={{ color: slider.color }}>
                  {values[slider.id]}/10
                </span>
              </div>
              <div className="playground-slider-track-wrapper">
                <input
                  type="range"
                  min={slider.min}
                  max={slider.max}
                  value={values[slider.id]}
                  onChange={e => handleSliderChange(slider.id, e.target.value)}
                  className="playground-range"
                  style={{ "--slider-color": slider.color, "--slider-pct": `${(values[slider.id] / slider.max) * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>

        {/* Prediction Cards */}
        <motion.div
          className="playground-results"
          key={JSON.stringify(values)}
          initial={{ opacity: 0.6, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.25 }}
        >
          <div className="playground-metrics">
            <article className="playground-metric-card">
              <span className="playground-metric-label">Comfort Score</span>
              <strong className="playground-metric-value" style={{ color: prediction.comfortScore >= 70 ? "var(--green)" : prediction.comfortScore >= 40 ? "var(--gold)" : "var(--red)" }}>
                {prediction.comfortScore}%
              </strong>
            </article>
            <article className="playground-metric-card">
              <span className="playground-metric-label">Adventure Rating</span>
              <strong className="playground-metric-value" style={{ color: "var(--violet)" }}>
                {prediction.adventureRating}%
              </strong>
            </article>
            <article className="playground-metric-card">
              <span className="playground-metric-label">Est. Cost</span>
              <strong className="playground-metric-value" style={{ color: "var(--cyan)" }}>
                ₹{prediction.costEstimate.toLocaleString("en-IN")}
              </strong>
            </article>
            <article className="playground-metric-card">
              <span className="playground-metric-label">Best Time</span>
              <strong className="playground-metric-value small">{prediction.optimalTime}</strong>
            </article>
          </div>

          {/* Persona Badge */}
          <div className="playground-persona-row">
            <div className="personality-badge">
              <span className="personality-badge-icon">
                {prediction.persona === "Adventure Seeker" ? "⚡" :
                 prediction.persona === "Budget Hacker" ? "💡" :
                 prediction.persona === "Luxury Traveler" ? "✨" : "🧭"}
              </span>
              <div>
                <strong>{prediction.persona}</strong>
                <small>Detected travel personality</small>
              </div>
            </div>
          </div>

          {/* AI Tips */}
          <div className="playground-tips">
            {prediction.tips.map((tip, i) => (
              <div key={i} className="playground-tip">
                <span className="playground-tip-icon">💡</span>
                <span>{tip}</span>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </FeatureShell>
  );
}

export default AIPlayground;
