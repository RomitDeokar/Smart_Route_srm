import { motion } from "framer-motion";
import { useMemo, useState } from "react";
import FeatureShell from "./FeatureShell.jsx";

const languageData = {
  default: {
    region: "General India",
    phrases: [
      { local: "Namaste", english: "Hello / Greetings", context: "Universal greeting" },
      { local: "Dhanyavaad", english: "Thank you", context: "Formal thanks" },
      { local: "Kitna?", english: "How much?", context: "Bargaining" },
      { local: "Haan / Nahi", english: "Yes / No", context: "Basic response" },
      { local: "Madat chahiye", english: "I need help", context: "Emergency" }
    ],
    etiquette: [
      "Remove shoes before entering temples and homes",
      "Use right hand for eating and passing items",
      "Avoid pointing feet at people or religious objects",
      "Bargaining is expected at local markets",
      "Dress modestly when visiting religious sites"
    ],
    tips: [
      "Carry small denominations for auto-rickshaws",
      "Download offline maps before remote travel",
      "Keep photocopies of important documents",
      "Drink bottled or filtered water",
      "Local SIM cards are affordable and widely available"
    ],
    currency: { symbol: "₹", name: "Indian Rupee (INR)", tipCustom: "5-10% at restaurants, round up for services" }
  },
  shillong: {
    region: "Meghalaya / Northeast",
    phrases: [
      { local: "Khublei", english: "Thank you (Khasi)", context: "Common in Shillong" },
      { local: "Kumno", english: "How are you? (Khasi)", context: "Casual greeting" },
      { local: "Phi long kumno?", english: "Where are you going?", context: "Common question" },
      { local: "Ka jingieid", english: "Love / Affection", context: "Cultural expression" },
      { local: "Nongkrem", english: "Traditional dance festival", context: "Cultural event" }
    ],
    etiquette: [
      "Meghalaya is a matrilineal society — respect local customs",
      "Ask permission before photographing locals",
      "Living root bridges require moderate fitness",
      "Carry rain gear year-round — Cherrapunji is nearby",
      "Support local Khasi cuisine — try Jadoh and Tungrymbai"
    ],
    tips: [
      "Police Bazaar is the commercial hub for shopping",
      "Shared Sumos are the local transport",
      "Book homestays for authentic experiences",
      "Dawki river is crystal clear Oct-April",
      "Carry cash — many spots don't accept cards"
    ],
    currency: { symbol: "₹", name: "Indian Rupee (INR)", tipCustom: "5-10% at restaurants, rounding up is appreciated" }
  },
  goa: {
    region: "Goa / Konkan",
    phrases: [
      { local: "Dev Borem Korum", english: "God bless (Konkani)", context: "Traditional greeting" },
      { local: "Koso asa?", english: "How are you? (Konkani)", context: "Casual" },
      { local: "Kitleak?", english: "How much? (Konkani)", context: "Markets" },
      { local: "Susegad", english: "Laid-back contentment", context: "Goan lifestyle" },
      { local: "Feni", english: "Local cashew/coconut spirit", context: "Cultural drink" }
    ],
    etiquette: [
      "Beach shacks are casual — flip-flops welcome",
      "Don't litter on beaches — heavy fines apply",
      "Wednesday afternoon is siesta time — shops may close",
      "Respect church timings and dress codes",
      "Taxi meters exist but negotiate before riding"
    ],
    tips: [
      "Rent a scooter for the best experience (₹300-500/day)",
      "North Goa = parties, South Goa = peace",
      "Try local fish thali at highway dhabas",
      "Visit Old Goa churches (UNESCO World Heritage)",
      "Best season: October to February"
    ],
    currency: { symbol: "₹", name: "Indian Rupee (INR)", tipCustom: "10% at restaurants, ₹50-100 for hotel staff" }
  }
};

function getLanguageForDestination(destination) {
  const dest = (destination || "").toLowerCase();
  if (dest.includes("shillong") || dest.includes("meghalaya") || dest.includes("cherrapunji")) return languageData.shillong;
  if (dest.includes("goa") || dest.includes("panaji") || dest.includes("calangute")) return languageData.goa;
  return languageData.default;
}

function LanguageAssistant({ destination }) {
  const data = useMemo(() => getLanguageForDestination(destination), [destination]);
  const [activeTab, setActiveTab] = useState("phrases");

  const tabs = [
    { id: "phrases", label: "Phrases", icon: "🗣️" },
    { id: "etiquette", label: "Etiquette", icon: "🙏" },
    { id: "tips", label: "Local Tips", icon: "💡" }
  ];

  return (
    <FeatureShell
      feature="Cultural"
      title="Language & Culture"
      subtitle={data.region}
      icon="🌍"
      defaultExpanded={false}
    >
      <div className="language-panel">
        {/* Currency Quick Ref */}
        <div className="language-currency-card">
          <span className="language-currency-symbol">{data.currency.symbol}</span>
          <div>
            <strong>{data.currency.name}</strong>
            <small>{data.currency.tipCustom}</small>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="language-tabs">
          {tabs.map(tab => (
            <button
              key={tab.id}
              className={`language-tab ${activeTab === tab.id ? "active" : ""}`}
              onClick={() => setActiveTab(tab.id)}
            >
              <span>{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.25 }}
          className="language-content"
        >
          {activeTab === "phrases" && (
            <div className="language-phrases">
              {data.phrases.map((phrase, i) => (
                <motion.div
                  key={i}
                  className="language-phrase-card"
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.06 }}
                >
                  <div className="phrase-local">{phrase.local}</div>
                  <div className="phrase-english">{phrase.english}</div>
                  <div className="phrase-context">{phrase.context}</div>
                </motion.div>
              ))}
            </div>
          )}

          {activeTab === "etiquette" && (
            <div className="language-list">
              {data.etiquette.map((item, i) => (
                <motion.div
                  key={i}
                  className="language-list-item"
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.06 }}
                >
                  <span className="language-list-icon">🙏</span>
                  <span>{item}</span>
                </motion.div>
              ))}
            </div>
          )}

          {activeTab === "tips" && (
            <div className="language-list">
              {data.tips.map((item, i) => (
                <motion.div
                  key={i}
                  className="language-list-item"
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.06 }}
                >
                  <span className="language-list-icon">💡</span>
                  <span>{item}</span>
                </motion.div>
              ))}
            </div>
          )}
        </motion.div>
      </div>
    </FeatureShell>
  );
}

export default LanguageAssistant;
