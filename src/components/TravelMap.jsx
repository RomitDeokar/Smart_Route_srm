import { useEffect, useMemo, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

const fallbackCenter = { latitude: 20.5937, longitude: 78.9629 };

function hashCode(value) {
  return String(value || "").split("").reduce((sum, char) => sum + char.charCodeAt(0), 0);
}

function getBaseCenter(destinationCoords, userLocation) {
  if (userLocation?.latitude && userLocation?.longitude) {
    return { latitude: userLocation.latitude, longitude: userLocation.longitude };
  }

  if (destinationCoords?.latitude && destinationCoords?.longitude) {
    return { latitude: destinationCoords.latitude, longitude: destinationCoords.longitude };
  }

  return fallbackCenter;
}

function buildAttractionPoints(attractions, baseCenter) {
  return (attractions || []).slice(0, 6).map((name, index) => {
    const hash = hashCode(`${name}-${index}`);
    const latOffset = (((hash % 17) - 8) * 0.012) + index * 0.004;
    const lonOffset = ((((hash >> 2) % 17) - 8) * 0.012) - index * 0.003;

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

function TravelMap({ destination, destinationCoords, userLocation, attractions }) {
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
        name: "Your location",
        type: "user",
        latitude: userLocation.latitude,
        longitude: userLocation.longitude
      });
    }

    entries.push({
      name: destination || "Destination hub",
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

  useEffect(() => {
    if (mapRef.current || !containerRef.current) {
      return undefined;
    }

    const map = L.map(containerRef.current, {
      zoomControl: true,
      attributionControl: true
    }).setView([baseCenter.latitude, baseCenter.longitude], 10);

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors"
    }).addTo(map);

    layerGroupRef.current = L.layerGroup().addTo(map);
    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
      layerGroupRef.current = null;
    };
  }, [baseCenter]);

  useEffect(() => {
    if (!mapRef.current || !layerGroupRef.current) {
      return;
    }

    const map = mapRef.current;
    const layerGroup = layerGroupRef.current;
    layerGroup.clearLayers();

    const markerIcon = type =>
      L.divIcon({
        className: "",
        html: `<span class="live-map-marker ${type}"></span>`,
        iconSize: [18, 18],
        iconAnchor: [9, 9]
      });

    points.forEach(point => {
      L.marker([point.latitude, point.longitude], { icon: markerIcon(point.type) })
        .bindPopup(
          `<div class="leaflet-popup-card"><strong>${point.name}</strong><span>${point.type}</span></div>`
        )
        .addTo(layerGroup);
    });

    if (points.length > 1) {
      const routeCoordinates = points.map(point => [point.latitude, point.longitude]);
      L.polyline(routeCoordinates, {
        color: "#73f0ff",
        weight: 4,
        opacity: 0.85
      }).addTo(layerGroup);
    }

    const bounds = L.latLngBounds(points.map(point => [point.latitude, point.longitude]));
    if (bounds.isValid()) {
      map.fitBounds(bounds, { padding: [48, 48], maxZoom: 12 });
    }
  }, [points]);

  return (
    <section className="glass-panel panel module-panel">
      <div className="section-heading compact">
        <div>
          <p className="eyebrow">Feature 6</p>
          <h3>Map Navigation</h3>
        </div>
      </div>

      <p className="panel-subtle">
        Interactive routed map with live user positioning, attraction markers, travel segments, and estimated movement time.
      </p>

      <div className="map-frame live-map-frame">
        <div ref={containerRef} className="live-map-canvas" />
      </div>

      <div className="map-meta-grid">
        <article className="map-meta-card">
          <strong>User location</strong>
          <span>{userLocation ? `${userLocation.latitude.toFixed(3)}, ${userLocation.longitude.toFixed(3)}` : "Waiting for geolocation"}</span>
        </article>
        <article className="map-meta-card">
          <strong>Estimated travel time</strong>
          <span>{routeStats.estimatedTravelTime}</span>
        </article>
      </div>

      <div className="map-meta-grid">
        <article className="map-meta-card">
          <strong>Route distance</strong>
          <span>{routeStats.totalDistance}</span>
        </article>
        <article className="map-meta-card">
          <strong>Displayed stops</strong>
          <span>{points.length}</span>
        </article>
      </div>

      <div className="map-marker-list">
        {points.map(point => (
          <span key={`${point.type}-${point.name}`}>{point.name}</span>
        ))}
      </div>

      <div className="route-segment-list">
        {routeStats.segments.map(segment => (
          <article key={`${segment.from}-${segment.to}`} className="route-segment-card">
            <strong>{segment.from} to {segment.to}</strong>
            <span>{segment.distance}</span>
            <span>{segment.travelTime}</span>
          </article>
        ))}
      </div>
    </section>
  );
}

export default TravelMap;
