import FeatureShell from "./FeatureShell.jsx";

function ItineraryPanel({ itinerary, loading, error, onGenerate }) {
  return (
    <FeatureShell
      feature="Feature 1"
      title="AI Itinerary Generator"
      subtitle="Structured day-by-day planning"
      icon="✦"
      loading={loading}
      error={error}
      action={
        <button className="button button-primary" type="button" onClick={onGenerate} disabled={loading}>
          {loading ? "Generating..." : "Generate Itinerary"}
        </button>
      }
    >
      <div className="itinerary-panel">
        <p className="panel-subtle">
          Generate a day-by-day travel plan using your current planner inputs.
        </p>

        {!itinerary ? (
          <div className="empty-feature-state">
            <strong>No itinerary yet</strong>
            <span>Click generate to see daily attractions, meals, and activities.</span>
          </div>
        ) : (
          <div className="generated-itinerary">
            <div className="itinerary-summary">
              <strong>{itinerary.title}</strong>
              <span>{itinerary.summary}</span>
            </div>

            <div className="itinerary-days">
              {itinerary.days.map(day => (
                <article key={day.day} className="itinerary-day-card">
                  <p className="tiny-label">Day {day.day}</p>
                  <h4>{day.theme}</h4>
                  <div className="itinerary-day-items">
                    {day.plan.map(item => (
                      <div key={`${day.day}-${item.type}-${item.name}`} className="itinerary-day-item">
                        <span className="item-tag">{item.type}</span>
                        <div>
                          <strong>{item.name}</strong>
                          <small>{item.note}</small>
                        </div>
                      </div>
                    ))}
                  </div>
                </article>
              ))}
            </div>
          </div>
        )}
      </div>
    </FeatureShell>
  );
}

export default ItineraryPanel;
