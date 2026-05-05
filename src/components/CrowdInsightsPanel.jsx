import { useState } from "react";

function CrowdInsightsPanel({ destination, attractions }) {
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
    <section className="glass-panel panel module-panel">
      <div className="section-heading compact">
        <div>
          <p className="eyebrow">Feature 9</p>
          <h3>Crowd & Best Time Prediction</h3>
        </div>
        <button className="button button-primary" type="button" onClick={fetchCrowdInfo} disabled={loading}>
          {loading ? "Analyzing..." : "Get Crowd Info"}
        </button>
      </div>

      {error ? <div className="feature-error">{error}</div> : null}

      <div className="module-card-grid">
        {results.map(location => (
          <article key={location.name} className="module-card">
            <strong>{location.name}</strong>
            <span>Peak hours: {location.peak_hours}</span>
            <span>Best time: {location.recommended_time}</span>
            <span>{location.indicator}</span>
          </article>
        ))}
      </div>
    </section>
  );
}

export default CrowdInsightsPanel;
