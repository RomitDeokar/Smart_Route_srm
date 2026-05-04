import { useState } from "react";
import FeatureShell from "./FeatureShell.jsx";

function EmergencyPanelPremium({ origin, destination }) {
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
    <FeatureShell
      feature="Feature 8"
      title="Emergency Travel Help"
      subtitle="Fallback options when plans change"
      icon="⚠"
      loading={loading}
      error={error}
      defaultExpanded={false}
      action={
        <button className="button button-primary" type="button" onClick={fetchEmergencyOptions} disabled={loading}>
          {loading ? "Searching..." : "Emergency Travel Help"}
        </button>
      }
    >
      <div className="module-panel">
        <p className="panel-subtle">
          Find nearby hotels, alternate flights, and transport options.
        </p>

        {data ? (
          <div className="emergency-grid">
            <article className="module-card premium-module-card">
              <strong>Nearby hotels</strong>
              {data.nearbyHotels.map(item => <span key={item}>{item}</span>)}
            </article>
            <article className="module-card premium-module-card">
              <strong>Alternate flights</strong>
              {data.alternateFlights.map(item => <span key={item}>{item}</span>)}
            </article>
            <article className="module-card premium-module-card">
              <strong>Transport options</strong>
              {data.transportOptions.map(item => <span key={item}>{item}</span>)}
            </article>
          </div>
        ) : null}
      </div>
    </FeatureShell>
  );
}

export default EmergencyPanelPremium;
