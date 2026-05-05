import { useEffect, useState } from "react";

function HotelsPanel({ defaultCity, defaultBudget }) {
  const [form, setForm] = useState({
    city: defaultCity,
    check_in: "",
    check_out: "",
    budget: defaultBudget
  });
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setForm(current => ({
      ...current,
      city: defaultCity,
      budget: defaultBudget
    }));
  }, [defaultCity, defaultBudget]);

  const searchHotels = async () => {
    setLoading(true);
    setError("");

    try {
      const response = await fetch("/api/hotels/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form)
      });

      const data = await response.json();
      if (!response.ok || !data.ok) {
        throw new Error(data.error || "Hotel search failed.");
      }

      setResults(data.hotels || []);
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
          <p className="eyebrow">Feature 5</p>
          <h3>Hotels Search</h3>
        </div>
        <button className="button button-primary" type="button" onClick={searchHotels} disabled={loading}>
          {loading ? "Searching..." : "Search Hotels"}
        </button>
      </div>

      <div className="module-form-grid">
        <label>
          <span>City</span>
          <input value={form.city} onChange={event => setForm({ ...form, city: event.target.value })} />
        </label>
        <label>
          <span>Check in</span>
          <input type="date" value={form.check_in} onChange={event => setForm({ ...form, check_in: event.target.value })} />
        </label>
        <label>
          <span>Check out</span>
          <input type="date" value={form.check_out} onChange={event => setForm({ ...form, check_out: event.target.value })} />
        </label>
        <label>
          <span>Budget</span>
          <input type="number" min="1000" step="500" value={form.budget} onChange={event => setForm({ ...form, budget: Number(event.target.value) || 0 })} />
        </label>
      </div>

      {error ? <div className="feature-error">{error}</div> : null}

      <div className="module-card-grid">
        {results.map(hotel => (
          <article key={`${hotel.name}-${hotel.price_per_night}`} className="module-card">
            <strong>{hotel.name}</strong>
            <span>{hotel.rating}★ rating</span>
            <span>{hotel.price_per_night} / night</span>
            <span>{hotel.distance_from_city_center} from center</span>
          </article>
        ))}
      </div>
    </section>
  );
}

export default HotelsPanel;
