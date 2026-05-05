import { useEffect, useState } from "react";

function PackingChecklistPanel({ defaultDestination }) {
  const [destination, setDestination] = useState(defaultDestination);
  const [travelDates, setTravelDates] = useState("");
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setDestination(defaultDestination);
  }, [defaultDestination]);

  const generateChecklist = async () => {
    setLoading(true);
    setError("");

    try {
      const response = await fetch("/api/packing-list", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          destination,
          travel_dates: travelDates
        })
      });

      const data = await response.json();
      if (!response.ok || !data.ok) {
        throw new Error(data.error || "Packing list generation failed.");
      }

      setItems(data.items || []);
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
          <p className="eyebrow">Feature 7</p>
          <h3>Packing Checklist</h3>
        </div>
        <button className="button button-primary" type="button" onClick={generateChecklist} disabled={loading}>
          {loading ? "Generating..." : "Generate Packing List"}
        </button>
      </div>

      <div className="module-form-grid">
        <label>
          <span>Destination</span>
          <input value={destination} onChange={event => setDestination(event.target.value)} />
        </label>
        <label>
          <span>Travel dates</span>
          <input value={travelDates} onChange={event => setTravelDates(event.target.value)} placeholder="e.g. Jun 10 - Jun 15" />
        </label>
      </div>

      {error ? <div className="feature-error">{error}</div> : null}

      <div className="checklist-grid">
        {items.map(item => (
          <label key={item} className="checklist-item">
            <input type="checkbox" />
            <span>{item}</span>
          </label>
        ))}
      </div>
    </section>
  );
}

export default PackingChecklistPanel;
