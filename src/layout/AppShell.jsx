import { useState } from "react";
import { NavLink, Outlet, useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import ChatFab from "../components/ChatFab.jsx";

/* ── Nav items — icons as SVG path strings, NOT JSX at module scope ── */
const NAV = [
  { to:"/",             label:"Dashboard",    exact:true,
    d:"M3 3h7v7H3zm11 0h7v7h-7zM3 14h7v7H3zm11 0h7v7h-7z" },
  { to:"/map",          label:"Map Explorer",
    d:"M1 6l7-4 8 4 7-4v16l-7 4-8-4-7 4z M8 2v16 M16 6v16" },
  { to:"/budget",       label:"Budget",
    d:"M12 1v22 M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" },
  { to:"/itinerary",    label:"Itinerary",
    d:"M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01" },
  { to:"/atlas",        label:"Travel Atlas",
    d:"M2 12a10 10 0 1 0 20 0a10 10 0 0 0-20 0z M2 12h20 M12 2a15.3 15.3 0 0 1 4 10a15.3 15.3 0 0 1-4 10a15.3 15.3 0 0 1-4-10a15.3 15.3 0 0 1 4-10z" },
  { to:"/packing",      label:"Smart Packing",
    d:"M5 7h14a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2z M9 7V5a3 3 0 0 1 6 0v2" },
  { to:"/ai",           label:"AI Assistant",
    d:"M13 2L3 14l9 0-1 8 10-12-9 0z" },
  { to:"/reservations", label:"Reservations",
    d:"M2 3h20a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z M8 21h8 M12 17v4" },
];

function NavIcon({ d }) {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      {d.split(" M").map((segment, i) => {
        const path = i === 0 ? segment : "M" + segment;
        return <path key={i} d={path} />;
      })}
    </svg>
  );
}

export default function AppShell({ tripCtx, setTripCtx, addToast, onLogout }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [search, setSearch]           = useState("");
  const navigate = useNavigate();

  const user = (() => {
    try { return JSON.parse(localStorage.getItem("sr_user")) || { name:"User" }; }
    catch { return { name:"User" }; }
  })();

  const handleSearch = (e) => {
    e.preventDefault();
    if (search.trim()) {
      setTripCtx(c => ({ ...c, destination: search.trim() }));
      addToast(`Searching trips to ${search.trim()}`, "info");
      setSearch("");
      navigate("/");
    }
  };

  return (
    <div className="app-shell">
      {/* ── Sidebar ── */}
      <aside className={`sidebar ${sidebarOpen ? "open" : ""}`}>
        <div className="sidebar-top">
          <div className="sidebar-brand">
            <div className="brand-logo">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
              </svg>
            </div>
            <div>
              <div className="brand-name">SmartRoute</div>
              <div className="brand-sub">SRMIST</div>
            </div>
          </div>
          <div style={{ fontSize:10.5, color:"var(--text-3)", fontWeight:600, letterSpacing:"0.06em", textTransform:"uppercase", paddingTop:2 }}>
            SRMIST Chapter
          </div>
        </div>

        <nav className="sidebar-nav">
          {NAV.map(({ to, label, d, exact }) => (
            <NavLink
              key={to} to={to} end={exact}
              className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}
              onClick={() => setSidebarOpen(false)}
            >
              <span className="nav-icon"><NavIcon d={d} /></span>
              {label}
            </NavLink>
          ))}
        </nav>

        <button
          className="sidebar-book-btn"
          onClick={() => { navigate("/reservations"); setSidebarOpen(false); }}
        >
          + Book New Trip
        </button>

        <div className="sidebar-bottom">
          <button className="nav-item" style={{ cursor:"pointer" }}>
            <span className="nav-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="3"/>
                <path d="M19.07 4.93a10 10 0 0 1 0 14.14M5.93 4.93a10 10 0 0 0 0 14.14"/>
              </svg>
            </span>
            Settings
          </button>
          <button className="nav-item" onClick={onLogout} style={{ cursor:"pointer", color:"var(--red)" }}>
            <span className="nav-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>
                <polyline points="16 17 21 12 16 7"/>
                <line x1="21" y1="12" x2="9" y2="12"/>
              </svg>
            </span>
            Sign out
          </button>
        </div>
      </aside>

      {/* ── Main content ── */}
      <div className="main-content">
        <header className="topbar">
          {/* Mobile menu (hidden on desktop, visible <=900px via .mobile-menu-btn) */}
          <button
            className="topbar-icon-btn mobile-menu-btn"
            onClick={() => setSidebarOpen(o => !o)}
            aria-label="Menu"
          >
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <line x1="3" y1="6" x2="21" y2="6"/>
              <line x1="3" y1="12" x2="21" y2="12"/>
              <line x1="3" y1="18" x2="21" y2="18"/>
            </svg>
          </button>

          {/* Nav links */}
          <div className="topbar-nav-links">
            <span className="topbar-nav-link">Destinations</span>
            <span className="topbar-nav-link">Itinerary</span>
            <span className="topbar-nav-link">Explore</span>
          </div>

          {/* Search */}
          <form className="topbar-search" onSubmit={handleSearch}>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color:"var(--text-3)", flexShrink:0 }}>
              <circle cx="11" cy="11" r="8"/>
              <line x1="21" y1="21" x2="16.65" y2="16.65"/>
            </svg>
            <input
              placeholder="Search SRMIST trips..."
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
          </form>

          <div className="topbar-right">
            <button className="topbar-icon-btn" title="Notifications">
              <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/>
                <path d="M13.73 21a2 2 0 0 1-3.46 0"/>
              </svg>
              <span className="notif-dot"/>
            </button>
            <div
              className="topbar-avatar"
              title={user.name}
              onClick={() => addToast(`Signed in as ${user.name}`, "info")}
            >
              {(user.initials || user.name?.[0] || "U").toUpperCase()}
            </div>
          </div>
        </header>

        <main className="page-content">
          <AnimatePresence mode="wait">
            <motion.div
              key={window.location.pathname}
              initial={{ opacity:0, y:10 }}
              animate={{ opacity:1, y:0 }}
              exit={{ opacity:0, y:-6 }}
              transition={{ duration:0.2, ease:"easeOut" }}
            >
              <Outlet />
            </motion.div>
          </AnimatePresence>
        </main>
      </div>

      <ChatFab tripCtx={tripCtx} addToast={addToast} />

      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          onClick={() => setSidebarOpen(false)}
          style={{ position:"fixed", inset:0, background:"rgba(0,0,0,0.35)", zIndex:99, backdropFilter:"blur(3px)" }}
        />
      )}
    </div>
  );
}
