import { useState } from "react";
import FeatureShell from "./FeatureShell.jsx";

function CrowdInsightsPanelPremium({ destination, attractions }) {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const fetchCrowdInfo = async () => {
    setLoading(true);
    setError("");

    try {
      const response = await fetch("/api/crowd-info", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          destination,
          attractions
        })
      });
      const data = await response.json();
      if (!response.ok || !data.ok) {
        throw new Error(data.error || "Crowd info fetch failed.");
      }
      setResults(data.locations || []);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <FeatureShell
      feature="Feature 9"
      title="Crowd & Best Time Prediction"
      subtitle="Optimal visit times for your attractions"
      icon="◍"
      loading={loading}
      error={error}
      defaultExpanded={false}
      action={
        <button className="button button-primary" type="button" onClick={fetchCrowdInfo} disabled={loading}>
          {loading ? "Analyzing..." : "Get Crowd Info"}
        </button>
      }
    >
      <div className="module-panel">
        <div className="module-card-grid">
          {results.map(location => (
            <article key={location.name} className="module-card premium-module-card">
              <strong>{location.name}</strong>
              <span>Peak hours: {location.peak_hours}</span>
              <span>Best time: {location.recommended_time}</span>
              <span>{location.indicator}</span>
            </article>
          ))}
        </div>
      </div>
    </FeatureShell>
  );
}

export default CrowdInsightsPanelPremium;
