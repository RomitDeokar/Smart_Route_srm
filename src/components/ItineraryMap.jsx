import { useEffect, useRef } from "react";

/* Renders ALL itinerary stops on a Leaflet map with day-coloured numbered
   markers, an origin → destination polyline, and click-to-open popups
   (with Google-Maps + Wikipedia links). */

const DAY_COLORS = ["#2563eb","#16a34a","#dc2626","#7c3aed","#ea580c","#0891b2","#db2777","#65a30d","#ca8a04","#0f766e"];

function ensureLeafletLoaded() {
  return new Promise((resolve) => {
    if (window.L) { resolve(window.L); return; }
    if (!document.getElementById("leaflet-css")) {
      const css = document.createElement("link");
      css.id = "leaflet-css"; css.rel = "stylesheet";
      css.href = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css";
      document.head.appendChild(css);
    }
    if (!document.getElementById("leaflet-js")) {
      const js = document.createElement("script");
      js.id = "leaflet-js";
      js.src = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js";
      js.onload = () => resolve(window.L);
      document.head.appendChild(js);
    } else {
      const wait = setInterval(() => {
        if (window.L) { clearInterval(wait); resolve(window.L); }
      }, 100);
    }
  });
}

export default function ItineraryMap({ itinerary, height = 420, activeDay = null }) {
  const mapRef = useRef(null);
  const mapObj = useRef(null);
  const layersRef = useRef([]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const L = await ensureLeafletLoaded();
      if (cancelled || !mapRef.current) return;
      if (mapObj.current) { mapObj.current.remove(); mapObj.current = null; }

      const center = itinerary?.destCoords ? [itinerary.destCoords.lat, itinerary.destCoords.lon] : [20.5937, 78.9629];
      const map = L.map(mapRef.current, {
        center, zoom: 11, zoomControl: true, attributionControl: false,
        preferCanvas: true, fadeAnimation: true,
      });
      mapObj.current = map;

      L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
        maxZoom: 19, subdomains: "abcd",
        detectRetina: true,
      }).addTo(map);

      drawAll(L);

      // Multiple invalidateSize calls to fix tile mis-rendering when map
      // is in a hidden tab or when parent layout shifts (mobile rotate, etc.)
      [60, 200, 450, 900, 1500].forEach(t => setTimeout(() => {
        try { mapObj.current && mapObj.current.invalidateSize(true); } catch {}
      }, t));
    })();
    // Window resize → recompute tiles
    const onResize = () => { try { mapObj.current && mapObj.current.invalidateSize(true); } catch {} };
    window.addEventListener("resize", onResize);
    return () => {
      cancelled = true;
      window.removeEventListener("resize", onResize);
      if (mapObj.current) { mapObj.current.remove(); mapObj.current = null; }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!mapObj.current || !window.L) return;
    drawAll(window.L);
    // After redraw, ensure tiles are sized correctly
    setTimeout(() => { try { mapObj.current && mapObj.current.invalidateSize(true); } catch {} }, 80);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [itinerary, activeDay]);

  function drawAll(L) {
    const map = mapObj.current; if (!map || !itinerary) return;

    layersRef.current.forEach(l => map.removeLayer(l));
    layersRef.current = [];

    const stops = itinerary.allStops || [];
    const filteredStops = activeDay ? stops.filter(s => s.day === activeDay) : stops;
    const allPoints = [];

    // Origin marker
    if (itinerary.originCoords) {
      const oIcon = L.divIcon({
        className: "",
        html: `<div style="width:14px;height:14px;border-radius:50%;background:#0ea5e9;border:3px solid white;box-shadow:0 2px 8px rgba(14,165,233,0.6)"></div>`,
        iconSize: [14,14], iconAnchor: [7,7],
      });
      const om = L.marker([itinerary.originCoords.lat, itinerary.originCoords.lon], { icon: oIcon })
        .bindTooltip(`<b>Origin:</b> ${itinerary.originCoords.name || itinerary.origin}`, { direction: "top" })
        .addTo(map);
      layersRef.current.push(om);
      allPoints.push([itinerary.originCoords.lat, itinerary.originCoords.lon]);
    }

    // Destination centre marker
    if (itinerary.destCoords) {
      const dIcon = L.divIcon({
        className: "",
        html: `<div style="width:18px;height:18px;border-radius:50%;background:#dc2626;border:3px solid white;box-shadow:0 3px 10px rgba(220,38,38,0.55)"></div>`,
        iconSize: [18,18], iconAnchor: [9,9],
      });
      const dm = L.marker([itinerary.destCoords.lat, itinerary.destCoords.lon], { icon: dIcon })
        .bindTooltip(`<b>${itinerary.destination}</b>`, { permanent: false, direction: "top" })
        .addTo(map);
      layersRef.current.push(dm);
      allPoints.push([itinerary.destCoords.lat, itinerary.destCoords.lon]);
    }

    // Origin → Destination dashed line
    if (itinerary.originCoords && itinerary.destCoords) {
      const line = L.polyline(
        [[itinerary.originCoords.lat, itinerary.originCoords.lon],
         [itinerary.destCoords.lat,   itinerary.destCoords.lon]],
        { color: "#0ea5e9", weight: 2, opacity: 0.55, dashArray: "8 8" }
      ).addTo(map);
      layersRef.current.push(line);
    }

    // Stops grouped by day with a connecting polyline per day
    const byDay = {};
    filteredStops.forEach(s => {
      if (!s.lat || !s.lon) return;
      (byDay[s.day] = byDay[s.day] || []).push(s);
    });

    Object.entries(byDay).forEach(([dayStr, list]) => {
      const day = Number(dayStr);
      const color = DAY_COLORS[(day - 1) % DAY_COLORS.length];
      list.forEach((s, idx) => {
        const html = `
          <div style="position:relative;width:30px;height:38px;">
            <div style="position:absolute;bottom:0;left:50%;transform:translateX(-50%);
                        width:0;height:0;border-left:7px solid transparent;border-right:7px solid transparent;border-top:10px solid ${color};"></div>
            <div style="position:absolute;top:0;left:0;width:30px;height:30px;border-radius:50%;
                        background:${color};color:white;display:flex;align-items:center;justify-content:center;
                        font-weight:800;font-size:12px;border:2.5px solid white;
                        box-shadow:0 3px 10px rgba(0,0,0,0.25);font-family:'Sora',sans-serif">D${day}</div>
          </div>`;
        const icon = L.divIcon({ className: "", html, iconSize: [30,38], iconAnchor: [15,38] });
        const popupHtml = `
          <div style="font-family:'Plus Jakarta Sans',sans-serif;min-width:200px">
            <div style="font-weight:700;font-size:13.5px;color:#111827;margin-bottom:4px">${s.name}</div>
            <div style="font-size:11.5px;color:${color};font-weight:600;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px">
              Day ${day} · ${s.type} · ${s.time || ""}
            </div>
            <div style="font-size:12px;color:#4b5563;line-height:1.5;margin-bottom:8px">${(s.note||"").slice(0,160)}</div>
            <div style="display:flex;gap:6px;flex-wrap:wrap">
              ${s.mapsUrl ? `<a href="${s.mapsUrl}" target="_blank" rel="noopener" style="font-size:11.5px;color:#2563eb;text-decoration:none;font-weight:600">Maps →</a>` : ""}
              ${s.wikiTitle ? `<a href="https://en.wikipedia.org/wiki/${s.wikiTitle}" target="_blank" rel="noopener" style="font-size:11.5px;color:#7c3aed;text-decoration:none;font-weight:600">Wiki →</a>` : ""}
              ${s.bookingUrl ? `<a href="${s.bookingUrl}" target="_blank" rel="noopener" style="font-size:11.5px;color:#16a34a;text-decoration:none;font-weight:600">Book →</a>` : ""}
            </div>
          </div>`;
        const m = L.marker([s.lat, s.lon], { icon })
          .bindPopup(popupHtml, { maxWidth: 260 })
          .bindTooltip(s.name, { direction: "top" })
          .addTo(map);
        layersRef.current.push(m);
        allPoints.push([s.lat, s.lon]);
      });
      // Day connector polyline
      if (list.length > 1) {
        const poly = L.polyline(list.map(s => [s.lat, s.lon]), { color, weight: 3, opacity: 0.6 }).addTo(map);
        layersRef.current.push(poly);
      }
    });

    if (allPoints.length > 1) {
      try {
        map.fitBounds(L.latLngBounds(allPoints), {
          padding: [50, 50], animate: true, maxZoom: 13,
        });
      } catch {}
    } else if (allPoints.length === 1) {
      map.setView(allPoints[0], 13);
    } else if (itinerary?.destCoords) {
      // No stops drawn yet — at least centre on destination
      map.setView([itinerary.destCoords.lat, itinerary.destCoords.lon], 12);
    }
  }

  return (
    <div style={{ width:"100%", height, borderRadius:"var(--r-lg)", overflow:"hidden", border:"1.5px solid var(--border)" }}>
      <div ref={mapRef} style={{ width:"100%", height:"100%" }} />
    </div>
  );
}
