import { useState } from "react";
import { motion } from "framer-motion";
import FeatureShell from "./FeatureShell.jsx";

const categoryIcons = {
  cultural: "🏛️",
  natural: "🌿",
  historic: "🏰",
  foods: "🍽️",
  sport: "⚽",
  amusements: "🎡",
  shopping: "🛍️",
  religious: "🕌",
  museum: "🎨",
  park: "🌳",
  scenic: "📸",
  default: "📍"
};

function getCategoryIcon(kinds) {
  if (!kinds) return categoryIcons.default;
  const lower = kinds.toLowerCase();
  for (const [key, icon] of Object.entries(categoryIcons)) {
    if (lower.includes(key)) return icon;
  }
  return categoryIcons.default;
}

function ActivitiesPanel({ destination, defaultExpanded = false }) {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState("all");

  const searchActivities = async () => {
    setLoading(true);
    setError("");

    try {
      const response = await fetch("/api/activities/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ destination })
      });

      const data = await response.json();
      if (!response.ok || !data.ok) {
        throw new Error(data.error || "Activity search failed.");
      }

      setResults(data.activities || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const categories = ["all", ...new Set(
    results.flatMap(a => (a.kinds || "").split(",").map(k => k.trim()).filter(Boolean))
  )].slice(0, 8);

  const filtered = filter === "all"
    ? results
    : results.filter(a => (a.kinds || "").toLowerCase().includes(filter.toLowerCase()));

  return (
    <FeatureShell
      feature="Feature 7"
      title="Activities & Attractions"
      subtitle="Discover culture, nature, food, and more nearby"
      icon="🗺️"
      loading={loading}
      error={error}
      defaultExpanded={defaultExpanded}
      action={
        <button className="button button-primary" type="button" onClick={searchActivities} disabled={loading}>
          {loading ? "Searching..." : `Discover in ${destination || "destination"}`}
        </button>
      }
    >
      <div className="module-panel">
        {results.length > 0 && (
          <>
            <div className="category-filters">
              {categories.map(cat => (
                <button
                  key={cat}
                  type="button"
                  className={`sort-btn ${filter === cat ? "active" : ""}`}
                  onClick={() => setFilter(cat)}
                >
                  {cat === "all" ? "All" : cat}
                </button>
              ))}
            </div>

            <div className="activities-grid">
              {filtered.map((activity, i) => (
                <motion.article
                  key={activity.id || `${activity.name}-${i}`}
                  className="activity-card"
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: i * 0.05 }}
                  whileHover={{ y: -4, scale: 1.02 }}
                >
                  <div className="activity-card-icon">
                    {getCategoryIcon(activity.kinds)}
                  </div>
                  <div className="activity-card-name">{activity.name}</div>
                  <div className="activity-card-category">{activity.kinds}</div>
                  <div className="activity-card-distance">{activity.distance}</div>
                  {activity.rating > 0 && (
                    <div style={{ color: "var(--gold)", fontSize: "0.8rem" }}>
                      {"★".repeat(Math.min(activity.rating, 5))} ({activity.rating})
                    </div>
                  )}
                </motion.article>
              ))}
            </div>
          </>
        )}

        {!loading && results.length === 0 && (
          <div className="empty-feature-state">
            <strong>No activities loaded yet</strong>
            <span>Click discover to find real attractions near {destination || "your destination"}.</span>
          </div>
        )}
      </div>
    </FeatureShell>
  );
}

export default ActivitiesPanel;
