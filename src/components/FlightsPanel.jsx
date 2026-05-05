import { useEffect, useState } from "react";

function FlightsPanel({ defaultOrigin, defaultDestination }) {
  const [form, setForm] = useState({
    origin: defaultOrigin,
    destination: defaultDestination,
    departure_date: "",
    return_date: "",
    passengers: 1
  });
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setForm(current => ({
      ...current,
      origin: defaultOrigin,
      destination: defaultDestination
    }));
  }, [defaultOrigin, defaultDestination]);

  const searchFlights = async () => {
    setLoading(true);
    setError("");

    try {
      const response = await fetch("/api/flights/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form)
      });

      const data = await response.json();
      if (!response.ok || !data.ok) {
        throw new Error(data.error || "Flight search failed.");
      }

      setResults(data.flights || []);
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
          <p className="eyebrow">Feature 4</p>
          <h3>Flights Search</h3>
        </div>
        <button className="button button-primary" type="button" onClick={searchFlights} disabled={loading}>
          {loading ? "Searching..." : "Search Flights"}
        </button>
      </div>

      <div className="module-form-grid">
        <label>
          <span>Origin</span>
          <input value={form.origin} onChange={event => setForm({ ...form, origin: event.target.value })} />
        </label>
        <label>
          <span>Destination</span>
          <input value={form.destination} onChange={event => setForm({ ...form, destination: event.target.value })} />
        </label>
        <label>
          <span>Departure</span>
          <input type="date" value={form.departure_date} onChange={event => setForm({ ...form, departure_date: event.target.value })} />
        </label>
        <label>
          <span>Return</span>
          <input type="date" value={form.return_date} onChange={event => setForm({ ...form, return_date: event.target.value })} />
        </label>
        <label>
          <span>Passengers</span>
          <input type="number" min="1" max="8" value={form.passengers} onChange={event => setForm({ ...form, passengers: Number(event.target.value) || 1 })} />
        </label>
      </div>

      {error ? <div className="feature-error">{error}</div> : null}

      <div className="module-card-grid">
        {results.map(flight => (
          <article key={`${flight.airline}-${flight.departure_time}-${flight.price}`} className="module-card">
            <strong>{flight.airline}</strong>
            <span>{flight.departure_time}</span>
            <span>{flight.duration}</span>
            <span>{flight.price}</span>
          </article>
        ))}
      </div>
    </section>
  );
}

export default FlightsPanel;
