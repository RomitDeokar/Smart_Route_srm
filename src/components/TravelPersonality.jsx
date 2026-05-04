import { motion } from "framer-motion";
import { useMemo } from "react";

const personalityTypes = {
  explorer: {
    icon: "🧭",
    title: "Explorer",
    tagline: "Curious minds, hidden paths",
    traits: ["Curiosity", "Flexibility", "Culture", "Discovery"],
    color: "var(--blue)",
    gradient: "linear-gradient(135deg, rgba(102,126,234,0.12), rgba(139,92,246,0.08))"
  },
  budget_hacker: {
    icon: "💡",
    title: "Budget Hacker",
    tagline: "Maximum value, minimum spend",
    traits: ["Resourceful", "Strategic", "Local Food", "Free Spots"],
    color: "var(--green)",
    gradient: "linear-gradient(135deg, rgba(16,185,129,0.12), rgba(110,231,183,0.08))"
  },
  luxury: {
    icon: "✨",
    title: "Luxury Traveler",
    tagline: "Premium comfort, curated experiences",
    traits: ["Comfort", "Exclusivity", "Fine Dining", "Premium Stay"],
    color: "var(--gold)",
    gradient: "linear-gradient(135deg, rgba(245,158,11,0.12), rgba(251,146,60,0.08))"
  },
  adventure: {
    icon: "⚡",
    title: "Adventure Seeker",
    tagline: "Thrill first, comfort second",
    traits: ["Adrenaline", "Nature", "Off-Road", "Spontaneous"],
    color: "var(--violet)",
    gradient: "linear-gradient(135deg, rgba(139,92,246,0.12), rgba(236,72,153,0.08))"
  }
};

function derivePersonality(persona, budget, services) {
  if (persona === "creator") return personalityTypes.explorer;
  if (persona === "student") return personalityTypes.budget_hacker;
  if (persona === "family") return personalityTypes.luxury;

  const serviceList = Array.isArray(services) ? services : [];
  const hasAdventure = serviceList.some(s =>
    s.toLowerCase().includes("event") || s.toLowerCase().includes("rain")
  );

  if (hasAdventure) return personalityTypes.adventure;
  if (budget > 25000) return personalityTypes.luxury;
  if (budget < 8000) return personalityTypes.budget_hacker;
  return personalityTypes.explorer;
}

function TravelPersonality({ persona, budget, services }) {
  const personality = useMemo(
    () => derivePersonality(persona, budget, services),
    [persona, budget, services]
  );

  return (
    <motion.div
      className="travel-personality-card"
      style={{ background: personality.gradient }}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      whileHover={{ y: -3, scale: 1.01 }}
    >
      <div className="personality-header">
        <motion.span
          className="personality-icon"
          animate={{ rotate: [0, 8, -8, 0] }}
          transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
        >
          {personality.icon}
        </motion.span>
        <div>
          <p className="eyebrow" style={{ color: personality.color }}>Your Travel Personality</p>
          <h4 className="personality-title">{personality.title}</h4>
          <p className="personality-tagline">{personality.tagline}</p>
        </div>
      </div>

      <div className="personality-traits">
        {personality.traits.map((trait, i) => (
          <motion.span
            key={trait}
            className="personality-trait"
            style={{ borderColor: personality.color + "30", color: personality.color }}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.1 * i, duration: 0.3 }}
          >
            {trait}
          </motion.span>
        ))}
      </div>

      {/* Trait radar visualization (CSS-only) */}
      <div className="personality-radar">
        {personality.traits.map((trait, i) => (
          <div
            key={`bar-${trait}`}
            className="personality-radar-bar"
            style={{ "--bar-color": personality.color }}
          >
            <span className="personality-radar-label">{trait}</span>
            <div className="personality-radar-track">
              <motion.div
                className="personality-radar-fill"
                initial={{ width: 0 }}
                animate={{ width: `${60 + (i * 10) + Math.random() * 15}%` }}
                transition={{ delay: 0.2 + i * 0.1, duration: 0.6, ease: "easeOut" }}
              />
            </div>
          </div>
        ))}
      </div>
    </motion.div>
  );
}

export default TravelPersonality;
