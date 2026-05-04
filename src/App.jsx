import { useState, useEffect } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import "./styles.css";
import AppShell     from "./layout/AppShell.jsx";
import Dashboard    from "./pages/Dashboard.jsx";
import MapExplorer  from "./pages/MapExplorer.jsx";
import Budget       from "./pages/Budget.jsx";
import Itinerary    from "./pages/Itinerary.jsx";
import AIAssistant  from "./pages/AIAssistant.jsx";
import Reservations from "./pages/Reservations.jsx";
import Atlas        from "./pages/Atlas.jsx";
import Packing      from "./pages/Packing.jsx";
import Login        from "./pages/Login.jsx";
import Register     from "./pages/Register.jsx";
import SplashScreen from "./components/SplashScreen.jsx";
import Toast        from "./components/Toast.jsx";

export default function App() {
  const [splash, setSplash]   = useState(true);
  const [toasts, setToasts]   = useState([]);
  const [authed, setAuthed]   = useState(() => !!localStorage.getItem("sr_user"));
  const [tripCtx, setTripCtx] = useState({
    origin:      "SRMIST Kattankulathur",
    destination: "Goa",
    days:        5,
    budget:      35000,
    persona:     "explorer",
    services:    ["Hotels","Food","Attractions","Language tips"]
  });

  useEffect(() => {
    const t = setTimeout(() => setSplash(false), 2200);
    return () => clearTimeout(t);
  }, []);

  const addToast = (msg, type = "info") => {
    const id = Date.now() + Math.random();
    setToasts(t => [...t, { id, msg, type }]);
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 4200);
  };

  const handleAuth = (user) => {
    localStorage.setItem("sr_user", JSON.stringify(user));
    setAuthed(true);
    addToast(`Welcome, ${user.name}!`, "success");
  };

  const handleLogout = () => {
    localStorage.removeItem("sr_user");
    setAuthed(false);
  };

  if (splash) return <SplashScreen />;

  return (
    <BrowserRouter>
      <Toast toasts={toasts} />
      <Routes>
        <Route path="/login"    element={!authed ? <Login    onAuth={handleAuth} /> : <Navigate to="/" />} />
        <Route path="/register" element={!authed ? <Register onAuth={handleAuth} /> : <Navigate to="/" />} />
        <Route path="/" element={
          authed
            ? <AppShell tripCtx={tripCtx} setTripCtx={setTripCtx} addToast={addToast} onLogout={handleLogout} />
            : <Navigate to="/login" />
        }>
          <Route index               element={<Dashboard    tripCtx={tripCtx} setTripCtx={setTripCtx} addToast={addToast} />} />
          <Route path="map"          element={<MapExplorer  tripCtx={tripCtx} addToast={addToast} />} />
          <Route path="budget"       element={<Budget       tripCtx={tripCtx} addToast={addToast} />} />
          <Route path="itinerary"    element={<Itinerary    tripCtx={tripCtx} addToast={addToast} />} />
          <Route path="ai"           element={<AIAssistant  tripCtx={tripCtx} addToast={addToast} />} />
          <Route path="reservations" element={<Reservations addToast={addToast} />} />
          <Route path="atlas"        element={<Atlas        tripCtx={tripCtx} addToast={addToast} />} />
          <Route path="packing"      element={<Packing      tripCtx={tripCtx} addToast={addToast} />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
