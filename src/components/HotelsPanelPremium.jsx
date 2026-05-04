import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import FeatureShell from "./FeatureShell.jsx";

function HotelsPanelPremium({ defaultCity, defaultBudget, defaultExpanded }) {
  const [form, setForm] = useState({
    city: defaultCity,
    check_in: "",
    check_out: "",
    budget: defaultBudget
  });
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [sortBy, setSortBy] = useState("price");
  const [globalBookingUrl, setGlobalBookingUrl] = useState("");

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
      setGlobalBookingUrl(data.bookingUrl || "");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  };

  const sorted = [...results].sort((a, b) => {
    if (sortBy === "price") return a.pricePerNight - b.pricePerNight;
    if (sortBy === "rating") return Number(b.rating) - Number(a.rating);
    if (sortBy === "distance") return parseFloat(a.distanceFromCenter) - parseFloat(b.distanceFromCenter);
    return 0;
  });

  const renderStars = (count) => {
    const full = Math.floor(count);
    const half = count % 1 >= 0.5;
    return "★".repeat(full) + (half ? "½" : "");
  };

  return (
    <FeatureShell
      feature="Feature 5"
      title="Hotels Search"
      subtitle="Compare stays with pricing and instant booking"
      icon="🏨"
      loading={loading}
      error={error}
      defaultExpanded={defaultExpanded !== undefined ? defaultExpanded : false}
    >
      <div className="module-panel">
        <div className="module-form-grid">
          <label>
            <span>City</span>
            <input value={form.city} onChange={e => setForm({ ...form, city: e.target.value })} placeholder="e.g. Shillong" />
          </label>
          <label>
            <span>Check in</span>
            <input type="date" value={form.check_in} onChange={e => setForm({ ...form, check_in: e.target.value })} />
          </label>
          <label>
            <span>Check out</span>
            <input type="date" value={form.check_out} onChange={e => setForm({ ...form, check_out: e.target.value })} />
          </label>
          <label>
            <span>Budget (₹)</span>
            <input type="number" min="1000" step="500" value={form.budget} onChange={e => setForm({ ...form, budget: Number(e.target.value) || 0 })} />
          </label>
        </div>

        <button className="button button-primary" type="button" onClick={searchHotels} disabled={loading} style={{ marginTop: "8px" }}>
          {loading ? "Searching..." : "Search Hotels"}
        </button>

        {results.length > 0 && (
          <>
            <div className="sort-controls">
              <span style={{ color: "var(--muted)", fontSize: "0.78rem", marginRight: "8px", alignSelf: "center" }}>Sort by:</span>
              {[["price", "Price"], ["rating", "Rating"], ["distance", "Distance"]].map(([key, label]) => (
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
              {sorted.map((hotel, i) => (
                <motion.article
                  key={`${hotel.name}-${i}`}
                  className="module-card premium-module-card"
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: i * 0.06 }}
                  whileHover={{ y: -4 }}
                >
                  <div className="hotel-card-top">
                    <div>
                      <div className="hotel-name">{hotel.name}</div>
                      <span style={{ color: "var(--muted)", fontSize: "0.75rem" }}>{hotel.chain}</span>
                    </div>
                    <div className="hotel-rating">
                      <span style={{ fontSize: "0.9rem" }}>{renderStars(hotel.stars)}</span>
                    </div>
                  </div>

                  <div className="hotel-price-row">
                    <span className="hotel-price">{hotel.priceFormatted}</span>
                    <span className="hotel-price-unit">/ night</span>
                  </div>

                  <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                    <span style={{ color: "var(--gold)", fontSize: "0.92rem", fontWeight: 700 }}>{hotel.rating}★</span>
                    <span style={{ color: "var(--muted)", fontSize: "0.78rem" }}>({hotel.reviewCount} reviews)</span>
                  </div>

                  <div className="hotel-amenities">
                    {hotel.amenities?.map(amenity => (
                      <span key={amenity} className="hotel-amenity">{amenity}</span>
                    ))}
                  </div>

                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span className="hotel-distance">📍 {hotel.distanceFromCenter} from center</span>
                    <span className="flight-tag" style={{ color: hotel.cancellationPolicy === "Free cancellation" ? "var(--green)" : "var(--muted)" }}>
                      {hotel.cancellationPolicy}
                    </span>
                  </div>

                  <a
                    href={hotel.bookingUrl || globalBookingUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="booking-btn"
                  >
                    Book on Google Hotels →
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

export default HotelsPanelPremium;
