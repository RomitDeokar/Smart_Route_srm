import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import FeatureShell from "./FeatureShell.jsx";

function FlightsPanelPremium({ defaultOrigin, defaultDestination, defaultExpanded }) {
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
  const [sortBy, setSortBy] = useState("price");
  const [globalBookingUrl, setGlobalBookingUrl] = useState("");

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
      setGlobalBookingUrl(data.bookingUrl || "");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  };

  const sorted = [...results].sort((a, b) => {
    if (sortBy === "price") return a.price - b.price;
    if (sortBy === "duration") return a.duration.localeCompare(b.duration);
    if (sortBy === "departure") return a.departureTime.localeCompare(b.departureTime);
    return 0;
  });

  return (
    <FeatureShell
      feature="Feature 4"
      title="Flights Search"
      subtitle="Real-time comparison with booking links"
      icon="✈️"
      loading={loading}
      error={error}
      defaultExpanded={defaultExpanded !== undefined ? defaultExpanded : false}
    >
      <div className="module-panel">
        <div className="module-form-grid">
          <label>
            <span>From</span>
            <input value={form.origin} onChange={e => setForm({ ...form, origin: e.target.value })} placeholder="e.g. Chennai" />
          </label>
          <label>
            <span>To</span>
            <input value={form.destination} onChange={e => setForm({ ...form, destination: e.target.value })} placeholder="e.g. Shillong" />
          </label>
          <label>
            <span>Departure</span>
            <input type="date" value={form.departure_date} onChange={e => setForm({ ...form, departure_date: e.target.value })} />
          </label>
          <label>
            <span>Return</span>
            <input type="date" value={form.return_date} onChange={e => setForm({ ...form, return_date: e.target.value })} />
          </label>
          <label>
            <span>Passengers</span>
            <input type="number" min="1" max="9" value={form.passengers} onChange={e => setForm({ ...form, passengers: Number(e.target.value) || 1 })} />
          </label>
        </div>

        <button className="button button-primary" type="button" onClick={searchFlights} disabled={loading} style={{ marginTop: "8px" }}>
          {loading ? "Searching..." : "Search Flights"}
        </button>

        {results.length > 0 && (
          <>
            <div className="sort-controls">
              <span style={{ color: "var(--muted)", fontSize: "0.78rem", marginRight: "8px", alignSelf: "center" }}>Sort by:</span>
              {[["price", "Price"], ["duration", "Duration"], ["departure", "Departure"]].map(([key, label]) => (
                <button
                  key={key}
                  type="button"
                  className={`sort-btn ${sortBy === key ? "active" : ""}`}
                  onClick={() => setSortBy(key)}
                >
                  {label}
                </button>
              ))}
            </div>

            <div className="module-card-grid">
              {sorted.map((flight, i) => (
                <motion.article
                  key={`${flight.flightNo}-${i}`}
                  className="module-card premium-module-card"
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: i * 0.06 }}
                  whileHover={{ y: -4 }}
                >
                  <div className="flight-card-header">
                    <div>
                      <div className="flight-airline">{flight.airline}</div>
                      <span style={{ color: "var(--muted)", fontSize: "0.75rem" }}>{flight.flightNo}</span>
                    </div>
                    <div className="flight-price">{flight.priceFormatted}</div>
                  </div>

                  <div className="flight-route">
                    <div>
                      <div className="flight-time">{flight.departureTime}</div>
                      <span className="flight-city">{form.origin}</span>
                    </div>
                    <div className="flight-connector">
                      <span className="flight-line" />
                      <span className="flight-duration-badge">{flight.duration}</span>
                      <span className="flight-line" />
                    </div>
                    <div style={{ textAlign: "right" }}>
                      <div className="flight-time">{flight.arrivalTime}</div>
                      <span className="flight-city">{form.destination}</span>
                    </div>
                  </div>

                  <div className="flight-meta">
                    <span className="flight-tag">{flight.stops}</span>
                    <span className="flight-tag">{flight.class}</span>
                    {flight.refundable && <span className="flight-tag" style={{ color: "var(--green)" }}>Refundable</span>}
                    {form.passengers > 1 && <span className="flight-tag">{flight.perPerson}/person</span>}
                  </div>

                  <a
                    href={flight.bookingUrl || globalBookingUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="booking-btn"
                  >
                    Book on Google Flights →
                  </a>
                </motion.article>
              ))}
            </div>
          </>
        )}
      </div>
    </FeatureShell>
  );
}

export default FlightsPanelPremium;
