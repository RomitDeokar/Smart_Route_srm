import { useState } from "react";

function EmergencyPanel({ origin, destination }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const fetchEmergencyOptions = async () => {
    setLoading(true);
    setError("");

    try {
      const response = await fetch("/api/emergency-options", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ origin, destination })
      });

      const result = await response.json();
      if (!response.ok || !result.ok) {
        throw new Error(result.error || "Emergency help failed.");
      }

      setData(result.options);
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
          <p className="eyebrow">Feature 8</p>
          <h3>Emergency Travel Help</h3>
        </div>
        <button className="button button-primary" type="button" onClick={fetchEmergencyOptions} disabled={loading}>
          {loading ? "Searching..." : "Emergency Travel Help"}
        </button>
      </div>

      <p className="panel-subtle">
        Find mock fallback hotels, alternate flights, and nearby transport options when plans break down.
      </p>

      {error ? <div className="feature-error">{error}</div> : null}

      {data ? (
        <div className="emergency-grid">
          <article className="module-card">
            <strong>Nearby hotels</strong>
            {data.nearbyHotels.map(item => <span key={item}>{item}</span>)}
          </article>
          <article className="module-card">
            <strong>Alternate flights</strong>
            {data.alternateFlights.map(item => <span key={item}>{item}</span>)}
          </article>
          <article className="module-card">
            <strong>Transport options</strong>
            {data.transportOptions.map(item => <span key={item}>{item}</span>)}
          </article>
        </div>
      ) : null}
    </section>
  );
}

export default EmergencyPanel;
