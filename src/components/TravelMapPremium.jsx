import { useEffect, useMemo, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import FeatureShell from "./FeatureShell.jsx";

const fallbackCenter = { latitude: 20.5937, longitude: 78.9629 };

function hashCode(value) {
  return String(value || "").split("").reduce((sum, char) => sum + char.charCodeAt(0), 0);
}

function getBaseCenter(destinationCoords, userLocation) {
  if (destinationCoords?.latitude && destinationCoords?.longitude) {
    return { latitude: destinationCoords.latitude, longitude: destinationCoords.longitude };
  }

  if (userLocation?.latitude && userLocation?.longitude) {
    return { latitude: userLocation.latitude, longitude: userLocation.longitude };
  }

  return fallbackCenter;
}

function buildAttractionPoints(attractions, baseCenter) {
  return (attractions || []).slice(0, 8).map((name, index) => {
    const hash = hashCode(`${name}-${index}`);
    const latOffset = (((hash % 17) - 8) * 0.01) + index * 0.003;
    const lonOffset = ((((hash >> 2) % 17) - 8) * 0.01) - index * 0.002;

    return {
      name,
      type: "attraction",
      latitude: baseCenter.latitude + latOffset,
      longitude: baseCenter.longitude + lonOffset
    };
  });
}

function haversineDistanceKm(start, end) {
  const toRad = degrees => (degrees * Math.PI) / 180;
  const earthRadius = 6371;
  const dLat = toRad(end.latitude - start.latitude);
  const dLon = toRad(end.longitude - start.longitude);
  const lat1 = toRad(start.latitude);
  const lat2 = toRad(end.latitude);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.sin(dLon / 2) ** 2 * Math.cos(lat1) * Math.cos(lat2);
  return 2 * earthRadius * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function TravelMapPremium({ destination, destinationCoords, userLocation, attractions }) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const layerGroupRef = useRef(null);

  const baseCenter = useMemo(
    () => getBaseCenter(destinationCoords, userLocation),
    [destinationCoords, userLocation]
  );

  const points = useMemo(() => {
    const entries = [];
    if (userLocation?.latitude && userLocation?.longitude) {
      entries.push({
        name: "📍 Your Location",
        type: "user",
        latitude: userLocation.latitude,
        longitude: userLocation.longitude
      });
    }

    entries.push({
      name: destination || "Destination",
      type: "destination",
      latitude: baseCenter.latitude,
      longitude: baseCenter.longitude
    });

    return [...entries, ...buildAttractionPoints(attractions, baseCenter)];
  }, [attractions, baseCenter, destination, userLocation]);

  const routeStats = useMemo(() => {
    const segments = [];
    let totalDistance = 0;
    for (let index = 1; index < points.length; index += 1) {
      const from = points[index - 1];
      const to = points[index];
      const distance = haversineDistanceKm(from, to);
      const travelMinutes = Math.max(8, Math.round((distance / 28) * 60));
      totalDistance += distance;
      segments.push({
        from: from.name,
        to: to.name,
        distance: `${distance.toFixed(1)} km`,
        travelTime: `${travelMinutes} mins`
      });
    }
    return {
      totalDistance: `${totalDistance.toFixed(1)} km`,
      estimatedTravelTime: `${Math.max(10, Math.round((totalDistance / 28) * 60))} mins`,
      segments
    };
  }, [points]);

  /* ── Initialize map ── */
  useEffect(() => {
    if (!containerRef.current) return undefined;

    // Destroy existing map if re-initializing
    if (mapRef.current) {
      mapRef.current.remove();
      mapRef.current = null;
      layerGroupRef.current = null;
    }

    const map = L.map(containerRef.current, {
      zoomControl: true,
      attributionControl: true,
      scrollWheelZoom: true
    }).setView([baseCenter.latitude, baseCenter.longitude], 10);

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors"
    }).addTo(map);

    layerGroupRef.current = L.layerGroup().addTo(map);
    mapRef.current = map;

    // Force a resize after mount to fix blank tile issue
    setTimeout(() => {
      map.invalidateSize();
    }, 300);

    return () => {
      map.remove();
      mapRef.current = null;
      layerGroupRef.current = null;
    };
  }, [baseCenter.latitude, baseCenter.longitude]);

  /* ── Update markers and polyline ── */
  useEffect(() => {
    if (!mapRef.current || !layerGroupRef.current) return;

    const map = mapRef.current;
    const layerGroup = layerGroupRef.current;
    layerGroup.clearLayers();

    const markerColors = {
      user: "#36e4a8",
      destination: "#f06caa",
      attraction: "#56d8f5"
    };

    const markerIcon = (type) =>
      L.divIcon({
        className: "",
        html: `<span class="live-map-marker ${type}" style="background:${markerColors[type] || "#56d8f5"}"></span>`,
        iconSize: [18, 18],
        iconAnchor: [9, 9]
      });

    points.forEach(point => {
      L.marker([point.latitude, point.longitude], { icon: markerIcon(point.type) })
        .bindPopup(`<div class="leaflet-popup-card"><strong>${point.name}</strong><span>${point.type}</span></div>`)
        .addTo(layerGroup);
    });

    if (points.length > 1) {
      L.polyline(points.map(point => [point.latitude, point.longitude]), {
        color: "#56d8f5",
        weight: 3,
        opacity: 0.75,
        dashArray: "8, 6"
      }).addTo(layerGroup);
    }

    const bounds = L.latLngBounds(points.map(point => [point.latitude, point.longitude]));
    if (bounds.isValid()) {
      map.fitBounds(bounds, { padding: [50, 50], maxZoom: 13 });
    }
  }, [points]);

  return (
    <FeatureShell
      feature="Feature 6"
      title="Interactive Map"
      subtitle="Live route visualization with attraction markers and travel segments"
      icon="🗺️"
      defaultExpanded={true}
    >
      <div className="module-panel">
        <div className="map-frame live-map-frame" style={{ minHeight: "420px" }}>
          <div ref={containerRef} className="live-map-canvas" style={{ height: "420px", width: "100%" }} />
        </div>

        <div className="map-meta-grid">
          <article className="map-meta-card">
            <strong>Destination</strong>
            <span>{destination || "Not set"} — {baseCenter.latitude.toFixed(3)}, {baseCenter.longitude.toFixed(3)}</span>
          </article>
          <article className="map-meta-card">
            <strong>User Location</strong>
            <span>{userLocation ? `${userLocation.latitude.toFixed(3)}, ${userLocation.longitude.toFixed(3)}` : "Awaiting GPS"}</span>
          </article>
        </div>

        <div className="map-meta-grid">
          <article className="map-meta-card">
            <strong>Route Distance</strong>
            <span style={{ fontFamily: "JetBrains Mono, monospace" }}>{routeStats.totalDistance}</span>
          </article>
          <article className="map-meta-card">
            <strong>Est. Travel Time</strong>
            <span style={{ fontFamily: "JetBrains Mono, monospace" }}>{routeStats.estimatedTravelTime}</span>
          </article>
        </div>

        {points.length > 0 && (
          <div className="map-marker-list">
            {points.map(point => (
              <span key={`${point.type}-${point.name}`}>
                {point.type === "user" ? "📍" : point.type === "destination" ? "🎯" : "⭐"} {point.name}
              </span>
            ))}
          </div>
        )}

        {routeStats.segments.length > 0 && (
          <div className="route-segment-list">
            {routeStats.segments.slice(0, 5).map(segment => (
              <article key={`${segment.from}-${segment.to}`} className="route-segment-card">
                <strong>{segment.from} → {segment.to}</strong>
                <span>{segment.distance} · {segment.travelTime}</span>
              </article>
            ))}
          </div>
        )}
      </div>
    </FeatureShell>
  );
}

export default TravelMapPremium;
