import FeatureShell from "./FeatureShell.jsx";

function QuickTripPanel({
  availableHours,
  onHoursChange,
  onFetch,
  loading,
  error,
  results,
  locationLabel
}) {
  return (
    <FeatureShell
      feature="Feature 2"
      title="Quick Trip Near Me"
      subtitle="Discover nearby spots using your location"
      icon="◎"
      loading={loading}
      error={error}
      action={
        <button className="button button-primary" type="button" onClick={onFetch} disabled={loading}>
          {loading ? "Locating..." : "Quick Trip Near Me"}
        </button>
      }
    >
      <div className="quick-trip-panel">
        <p className="panel-subtle">
          Find nearby places worth visiting when you have a few hours free.
        </p>

        <label className="quick-trip-hours">
          <span>Available hours</span>
          <input
            type="number"
            min="1"
            max="12"
            value={availableHours}
            onChange={event => onHoursChange(Number(event.target.value))}
          />
        </label>

        <div className="quick-trip-location">
          <strong>Location status</strong>
          <span>{locationLabel || "Waiting for your browser location."}</span>
        </div>

        {!results.length ? (
          <div className="empty-feature-state">
            <strong>No quick trip suggestions yet</strong>
            <span>Use your browser location to find nearby travel ideas.</span>
          </div>
        ) : (
          <div className="quick-trip-results">
            {results.map(place => (
              <article key={place.name} className="quick-trip-card">
                <div className="quick-trip-card-head">
                  <strong>{place.name}</strong>
                  <span className="pill">{place.rating}★</span>
                </div>

                <div className="quick-trip-metrics">
                  <div>
                    <span>Distance</span>
                    <strong>{place.distance}</strong>
                  </div>
                  <div>
                    <span>Travel Time</span>
                    <strong>{place.estimated_travel_time}</strong>
                  </div>
                </div>
              </article>
            ))}
          </div>
        )}
      </div>
    </FeatureShell>
  );
}

export default QuickTripPanel;
