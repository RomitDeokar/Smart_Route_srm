import { useEffect, useRef } from "react";

const CITY_COORDS = {
  shillong:  [25.5788, 91.8933],
  goa:       [15.2993, 74.124],
  ooty:      [11.4102, 76.695],
  munnar:    [10.0889, 77.0595],
  rishikesh: [30.0869, 78.2676],
  udaipur:   [24.5854, 73.7125],
  jaipur:    [26.9124, 75.7873],
  delhi:     [28.6139, 77.209],
  mumbai:    [19.076,  72.8777],
  chennai:   [13.0827, 80.2707],
  bangalore: [12.9716, 77.5946],
  kolkata:   [22.5726, 88.3639],
  manali:    [32.2396, 77.1887],
  shimla:    [31.1048, 77.1734],
  varanasi:  [25.3176, 83.0064],
  kochi:     [9.9312,  76.2673],
  pondicherry:[11.9416,79.8083],
  hampi:     [15.335,  76.462],
  varkala:   [8.7378,  76.7163],
};

function getCityCoords(name) {
  const key = (name || "").toLowerCase().split(",")[0].trim();
  for (const [city, coords] of Object.entries(CITY_COORDS)) {
    if (key.includes(city) || city.includes(key)) return coords;
  }
  return [20.5937, 78.9629]; // center of India
}

export default function MapView({ destination, origin, height = 290, className = "" }) {
  const mapRef     = useRef(null);
  const mapObjRef  = useRef(null);
  const layersRef  = useRef([]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (!window.L) {
      // Dynamically load Leaflet CSS + JS
      const link = document.createElement("link");
      link.rel  = "stylesheet";
      link.href = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css";
      document.head.appendChild(link);

      const script = document.createElement("script");
      script.src = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js";
      script.onload = () => initMap();
      document.head.appendChild(script);
    } else {
      initMap();
    }

    function initMap() {
      if (!mapRef.current || mapObjRef.current) return;
      const L = window.L;

      const destCoords   = getCityCoords(destination);
      const originCoords = getCityCoords(origin);

      const map = L.map(mapRef.current, {
        center: destCoords,
        zoom: 8,
        zoomControl: false,
        attributionControl: false,
      });
      mapObjRef.current = map;

      // Dark tile layer
      L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
        maxZoom: 19,
      }).addTo(map);

      // Custom icon
      const blueIcon = L.divIcon({
        className: "",
        html: `<div style="width:12px;height:12px;border-radius:50%;background:#3b82f6;border:2px solid #fff;box-shadow:0 0 8px rgba(59,130,246,0.6)"></div>`,
        iconSize: [12, 12],
        iconAnchor: [6, 6],
      });
      const orangeIcon = L.divIcon({
        className: "",
        html: `<div style="width:10px;height:10px;border-radius:50%;background:#f59e0b;border:2px solid #fff;box-shadow:0 0 8px rgba(245,158,11,0.5)"></div>`,
        iconSize: [10, 10],
        iconAnchor: [5, 5],
      });

      const destMarker = L.marker(destCoords, { icon: blueIcon })
        .bindPopup(`<b>${destination}</b><br>Destination`)
        .addTo(map);

      const origMarker = L.marker(originCoords, { icon: orangeIcon })
        .bindPopup(`<b>${origin}</b><br>Origin`)
        .addTo(map);

      // Dashed route line
      const line = L.polyline([originCoords, destCoords], {
        color: "#3b82f6",
        weight: 2,
        opacity: 0.6,
        dashArray: "6 8",
      }).addTo(map);

      layersRef.current = [destMarker, origMarker, line];

      // Fit bounds
      map.fitBounds([originCoords, destCoords], { padding: [30, 30] });
    }

    return () => {
      if (mapObjRef.current) {
        mapObjRef.current.remove();
        mapObjRef.current = null;
      }
    };
  }, []);

  // Update when destination/origin changes
  useEffect(() => {
    if (!mapObjRef.current || !window.L) return;
    const L = window.L;
    const map = mapObjRef.current;

    // Remove old layers
    layersRef.current.forEach(l => map.removeLayer(l));
    layersRef.current = [];

    const destCoords   = getCityCoords(destination);
    const originCoords = getCityCoords(origin);

    const blueIcon = L.divIcon({
      className: "",
      html: `<div style="width:12px;height:12px;border-radius:50%;background:#3b82f6;border:2px solid #fff;box-shadow:0 0 8px rgba(59,130,246,0.6)"></div>`,
      iconSize: [12, 12], iconAnchor: [6, 6],
    });
    const orangeIcon = L.divIcon({
      className: "",
      html: `<div style="width:10px;height:10px;border-radius:50%;background:#f59e0b;border:2px solid #fff;box-shadow:0 0 8px rgba(245,158,11,0.5)"></div>`,
      iconSize: [10, 10], iconAnchor: [5, 5],
    });

    const d = L.marker(destCoords, { icon: blueIcon }).addTo(map);
    const o = L.marker(originCoords, { icon: orangeIcon }).addTo(map);
    const line = L.polyline([originCoords, destCoords], { color:"#3b82f6", weight:2, opacity:0.6, dashArray:"6 8" }).addTo(map);

    layersRef.current = [d, o, line];
    map.fitBounds([originCoords, destCoords], { padding: [30, 30] });
  }, [destination, origin]);

  return (
    <div
      ref={mapRef}
      className={className}
      style={{ width: "100%", height: `${height}px`, borderRadius: "inherit" }}
    />
  );
}
