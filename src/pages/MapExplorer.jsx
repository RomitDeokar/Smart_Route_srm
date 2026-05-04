import { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";

const COORDS = {
  shillong:[25.5788,91.8933],goa:[15.2993,74.124],ooty:[11.4102,76.695],munnar:[10.0889,77.0595],
  rishikesh:[30.0869,78.2676],udaipur:[24.5854,73.7125],jaipur:[26.9124,75.7873],delhi:[28.6139,77.209],
  mumbai:[19.076,72.8777],chennai:[13.0827,80.2707],bangalore:[12.9716,77.5946],manali:[32.2396,77.1887],
  hampi:[15.335,76.462],pondicherry:[11.9416,79.8083],varanasi:[25.3176,83.0064],kochi:[9.9312,76.2673],
};
function getCoords(n){const k=(n||"").toLowerCase().split(",")[0].trim();for(const[c,coords]of Object.entries(COORDS)){if(k.includes(c)||c.includes(k))return coords;}return[20.5937,78.9629];}

export default function MapExplorer({ tripCtx, addToast }) {
  const [dest, setDest]        = useState(tripCtx.destination);
  const [activities, setActs]  = useState([]);
  const [loading, setLoading]  = useState(false);
  const [hours, setHours]      = useState(4);
  const [quickTrips, setQT]    = useState([]);
  const mapRef = useRef(null);
  const mapObj = useRef(null);

  useEffect(() => {
    const el = mapRef.current;
    if (!el || mapObj.current) return;
    const load = () => {
      if (window.L) { init(); return; }
      const css = document.createElement("link"); css.rel="stylesheet"; css.href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"; document.head.appendChild(css);
      const js = document.createElement("script"); js.src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"; js.onload=init; document.head.appendChild(js);
    };
    function init() {
      const L = window.L;
      const c = getCoords(dest);
      const map = L.map(el,{center:c,zoom:8,zoomControl:false,attributionControl:false});
      L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", { maxZoom:19, subdomains:"abcd" }).addTo(map);
      const icon = L.divIcon({className:"",html:`<div style="width:13px;height:13px;border-radius:50%;background:#2563eb;border:2.5px solid white;box-shadow:0 2px 8px rgba(37,99,235,0.5)"></div>`,iconSize:[13,13],iconAnchor:[6,6]});
      L.marker(c,{icon}).bindTooltip(dest,{permanent:true,direction:"top"}).addTo(map);
      mapObj.current = map;
    }
    load();
    return () => { if(mapObj.current){mapObj.current.remove();mapObj.current=null;} };
  }, []);

  useEffect(() => {
    if (!mapObj.current || !window.L) return;
    const L = window.L; const map = mapObj.current;
    map.eachLayer(l=>{ if(l instanceof L.Marker||l instanceof L.Polyline) map.removeLayer(l); });
    const c = getCoords(dest);
    const icon = L.divIcon({className:"",html:`<div style="width:13px;height:13px;border-radius:50%;background:#2563eb;border:2.5px solid white;box-shadow:0 2px 8px rgba(37,99,235,0.5)"></div>`,iconSize:[13,13],iconAnchor:[6,6]});
    L.marker(c,{icon}).bindTooltip(dest,{permanent:true,direction:"top"}).addTo(map);
    map.flyTo(c, 8, { duration:1.2 });
  }, [dest]);

  const fetchActs = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/activities/search",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({destination:dest})});
      const d = await res.json();
      if(d.ok){setActs(d.activities?.slice(0,8)||[]);addToast(`${d.activities?.length} activities found!`,"success");}
      // Add markers to map
      if(d.activities?.length && mapObj.current && window.L){
        const L = window.L;
        d.activities.slice(0,6).forEach((act,i)=>{
          if(act.point){
            const sm = L.divIcon({className:"",html:`<div style="width:9px;height:9px;border-radius:50%;background:#16a34a;border:2px solid white;box-shadow:0 1px 4px rgba(0,0,0,0.3)"></div>`,iconSize:[9,9],iconAnchor:[4,4]});
            L.marker([act.point.lat,act.point.lon],{icon:sm}).bindPopup(act.name).addTo(mapObj.current);
          }
        });
      }
    } catch(e){addToast(e.message,"error");}
    finally{setLoading(false);}
  };

  const fetchQT = async () => {
    if(!navigator.geolocation){addToast("Geolocation not available","error");return;}
    navigator.geolocation.getCurrentPosition(async pos=>{
      try{
        const res=await fetch("/api/quick-trip",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({latitude:pos.coords.latitude,longitude:pos.coords.longitude,available_hours:hours})});
        const d=await res.json(); if(d.ok){setQT(d.places||[]);addToast(`${d.places?.length} quick trips!`,"success");}
      }catch(e){addToast(e.message,"error");}
    },()=>addToast("Location access denied","error"),{enableHighAccuracy:true,timeout:8000});
  };

  return (
    <div>
      <div className="section-title">Map Explorer</div>
      <div className="section-sub">Interactive route planning with live activity data</div>
      <div className="map-explorer-layout">
        <div className="map-explorer-panel">
          {/* Search */}
          <div className="card">
            <div className="card-body" style={{display:"flex",flexDirection:"column",gap:10}}>
              <div className="field-group">
                <label className="field-label">Destination</label>
                <input className="field-input" value={dest} onChange={e=>setDest(e.target.value)} placeholder="City name..." />
              </div>
              <button className="btn btn-primary w-full" onClick={fetchActs} disabled={loading}>
                {loading ? "Searching..." : "Find Activities"}
              </button>
            </div>
          </div>

          {/* Quick trip */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Quick Trip</span>
              <span className="pill blue">{hours}h</span>
            </div>
            <div className="card-body" style={{display:"flex",flexDirection:"column",gap:10}}>
              <div className="field-group">
                <label className="field-label">Available time: {hours} hours</label>
                <input type="range" min={1} max={12} value={hours} onChange={e=>setHours(+e.target.value)} style={{width:"100%",accentColor:"var(--blue)"}} />
              </div>
              <button className="btn btn-ghost w-full" onClick={fetchQT}>Use My Location</button>
              {quickTrips.map((qt,i)=>(
                <motion.div key={i} initial={{opacity:0,y:4}} animate={{opacity:1,y:0}} transition={{delay:i*0.05}}
                  style={{background:"var(--bg-soft)",border:"1.5px solid var(--border)",borderRadius:"var(--r-md)",padding:"10px 12px"}}>
                  <div style={{fontSize:13,fontWeight:600,color:"var(--text)",marginBottom:3}}>{qt.name}</div>
                  <div style={{fontSize:11.5,color:"var(--text-3)"}}>{qt.distance} · {qt.estimated_travel_time} · {qt.rating}★</div>
                </motion.div>
              ))}
            </div>
          </div>

          {/* Activities */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Nearby</span>
              {activities.length>0&&<span className="pill green">{activities.length}</span>}
            </div>
            <div className="card-body" style={{display:"flex",flexDirection:"column",gap:8}}>
              {activities.length===0
                ? <p style={{color:"var(--text-3)",fontSize:13}}>Search a destination to discover activities.</p>
                : activities.map((act,i)=>(
                  <motion.div key={i} initial={{opacity:0,x:-6}} animate={{opacity:1,x:0}} transition={{delay:i*0.05}}
                    style={{background:"var(--bg-soft)",border:"1.5px solid var(--border)",borderRadius:"var(--r-md)",padding:"10px 12px",cursor:"pointer",transition:"all 0.12s"}}
                    whileHover={{borderColor:"var(--blue-border)",background:"white"}}>
                    <div style={{fontSize:13,fontWeight:600,color:"var(--text)",marginBottom:2}}>{act.name}</div>
                    <div style={{fontSize:11.5,color:"var(--text-3)"}}>{act.kinds?.split(",").slice(0,2).join(" · ")} {act.distance&&`· ${act.distance}`}</div>
                  </motion.div>
                ))
              }
            </div>
          </div>
        </div>

        <div className="map-explorer-main">
          <div className="live-badge"><span className="dot-live" />LIVE MAP</div>
          <div ref={mapRef} style={{width:"100%",height:"100%",minHeight:400}} />
          <div className="map-controls">
            <button className="map-ctrl-btn" onClick={()=>mapObj.current?.zoomIn()}>+</button>
            <button className="map-ctrl-btn" onClick={()=>mapObj.current?.zoomOut()}>−</button>
          </div>
        </div>
      </div>
    </div>
  );
}
