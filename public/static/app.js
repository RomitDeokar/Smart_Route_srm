// ============================================
// SmartRoute SRMIST — Agentic AI Travel Planner
// Complete Frontend — FREE APIs only (no OpenAI/Claude)
// APIs: OpenMeteo, Overpass/OSM, OpenTripMap, Wikipedia, Nominatim
// ============================================

const API_BASE = window.location.origin;

// === STATE ===
const state = {
  theme: localStorage.getItem('sr-theme') || 'light',
  persona: 'solo',
  itinerary: null,
  agents: {},
  logs: [],
  rl: { rewards: [], episode: 0 },
  bayesian: { cultural:{a:2,b:2}, adventure:{a:2,b:2}, food:{a:3,b:1}, relaxation:{a:1,b:3}, shopping:{a:1,b:2}, nature:{a:2,b:2}, nightlife:{a:1,b:3} },
  dirichlet: {},
  pomdpBelief: {},
  budget: { total:15000, used:0, breakdown:{} },
  chatOpen: false,
  generating: false,
  currentDest: '',
  currentOrigin: '',
  map: null,
  markers: [],
  routeLines: [],
  showRoutes: true,
  mapLayer: 'light',
  bookingCart: { flights:null, trains:null, hotels:null, cabs:null },
  bookingHistory: JSON.parse(localStorage.getItem('sr-history')||'[]'),
  packingList: {},
  packingChecked: JSON.parse(localStorage.getItem('sr-packing')||'{}'),
  atlasTrips: JSON.parse(localStorage.getItem('sr-atlas')||'[]'),
  rlChart: null,
  budgetChart: null,
  atlasMap: null,
};

// === AGENT DEFINITIONS ===
const AGENTS = [
  {id:'planner',name:'Planner Agent',role:'MCTS Itinerary Optimization',icon:'🗺️',color:'#667eea'},
  {id:'weather',name:'Weather Risk Agent',role:'Naive Bayes Weather Classification',icon:'🌦️',color:'#06b6d4'},
  {id:'crowd',name:'Crowd Analyzer',role:'Time-Based Crowd Prediction',icon:'👥',color:'#f59e0b'},
  {id:'budget',name:'Budget Optimizer',role:'MDP Budget Adherence',icon:'💰',color:'#10b981'},
  {id:'preference',name:'Preference Agent',role:'Bayesian Beta Learning',icon:'❤️',color:'#ec4899'},
  {id:'booking',name:'Booking Assistant',role:'Multi-Platform Search',icon:'🎫',color:'#8b5cf6'},
  {id:'explain',name:'Explainability Agent',role:'MDP Trace & POMDP Belief',icon:'🧠',color:'#f97316'},
];

// === CITY COORDINATES ===
const CITY_COORDS = {
  paris:[48.8566,2.3522],london:[51.5074,-0.1278],tokyo:[35.6762,139.6503],jaipur:[26.9124,75.7873],
  rome:[41.9028,12.4964],'new york':[40.7128,-74.006],dubai:[25.2048,55.2708],singapore:[1.3521,103.8198],
  bangkok:[13.7563,100.5018],chennai:[13.0827,80.2707],srm:[12.8231,80.0442],srmist:[12.8231,80.0442],
  mumbai:[19.076,72.8777],delhi:[28.7041,77.1025],bangalore:[12.9716,77.5946],hyderabad:[17.385,78.4867],
  kolkata:[22.5726,88.3639],goa:[15.2993,74.124],udaipur:[24.5854,73.7125],varanasi:[25.3176,83.0068],
  agra:[27.1767,78.0081],kochi:[9.9312,76.2673],shimla:[31.1048,77.1734],manali:[32.2432,77.1892],
  pondicherry:[11.9416,79.8083],mahabalipuram:[12.6169,80.1993],ooty:[11.41,76.695],mysore:[12.2958,76.6394],
  rishikesh:[30.0869,78.2676],darjeeling:[27.041,88.2663],amritsar:[31.634,74.8723],jodhpur:[26.2389,73.0243],
  leh:[34.1526,77.5771],munnar:[10.0889,77.0595],kodaikanal:[10.2381,77.4892],hampi:[15.335,76.46],
};

// === PHOTO CACHE & FALLBACKS ===
const _photoCache = new Map();
// Fallback is a neutral placeholder, NOT a specific place photo
const PLACEHOLDER_IMG = 'data:image/svg+xml,' + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300" fill="none"><rect width="400" height="300" fill="%23e5e7eb"/><text x="200" y="140" text-anchor="middle" fill="%239ca3af" font-family="system-ui" font-size="24">Loading photo...</text><text x="200" y="170" text-anchor="middle" fill="%239ca3af" font-family="system-ui" font-size="40">📷</text></svg>');

function getFallbackPhoto(type) {
  return PLACEHOLDER_IMG;
}

async function fetchPlacePhoto(name, type, wikiTitle) {
  const key = (wikiTitle || name).toLowerCase();
  if (_photoCache.has(key)) return _photoCache.get(key);
  
  // Try wikiTitle first (most accurate), then place name, then search
  const attempts = [wikiTitle || name, name];
  for (const title of [...new Set(attempts)]) {
    if (!title) continue;
    try {
      const r = await fetch(`https://en.wikipedia.org/w/api.php?action=query&format=json&titles=${encodeURIComponent(title)}&prop=pageimages&piprop=thumbnail&pithumbsize=400&redirects=1&origin=*`);
      const d = await r.json();
      const pages = d?.query?.pages || {};
      for (const p of Object.values(pages)) {
        if (p.thumbnail?.source && !p.thumbnail.source.includes('.svg') && !p.thumbnail.source.includes('Flag_of')) {
          _photoCache.set(key, p.thumbnail.source);
          return p.thumbnail.source;
        }
      }
    } catch(e) {}
  }
  
  // Try Wikipedia search API as last resort
  try {
    const r = await fetch(`https://en.wikipedia.org/w/api.php?action=query&format=json&generator=search&gsrsearch=${encodeURIComponent(name)}&gsrlimit=3&prop=pageimages&piprop=thumbnail&pithumbsize=400&origin=*`);
    const d = await r.json();
    const pages = d?.query?.pages || {};
    for (const p of Object.values(pages)) {
      if (p.thumbnail?.source && !p.thumbnail.source.includes('.svg') && !p.thumbnail.source.includes('Flag_of')) {
        _photoCache.set(key, p.thumbnail.source);
        return p.thumbnail.source;
      }
    }
  } catch(e) {}
  
  _photoCache.set(key, PLACEHOLDER_IMG);
  return PLACEHOLDER_IMG;
}

// ============================================
// INITIALIZATION
// ============================================
document.addEventListener('DOMContentLoaded', () => {
  applyTheme();
  initMap();
  renderAgentCards();
  checkBackend();
  updateClocks();
  setInterval(updateClocks, 1000);
  document.getElementById('startDate').valueAsDate = new Date();
  // Load atlas
  updateAtlasStats();
});

function applyTheme() {
  document.documentElement.setAttribute('data-theme', state.theme);
  const icon = document.getElementById('themeIcon');
  if (icon) icon.className = state.theme === 'dark' ? 'fas fa-moon' : 'fas fa-sun';
  // Also sync map layer with theme
  if (state.map && state.mapLayer !== (state.theme === 'dark' ? 'dark' : 'light')) {
    state.mapLayer = state.theme === 'dark' ? 'dark' : 'light';
    setMapLayer();
  }
}
function toggleTheme() {
  state.theme = state.theme === 'dark' ? 'light' : 'dark';
  localStorage.setItem('sr-theme', state.theme);
  applyTheme();
  if (state.map) setTimeout(() => state.map.invalidateSize(), 100);
}

// ============================================
// MAP (Leaflet — FREE)
// ============================================
function initMap() {
  state.map = L.map('map', { zoomControl: false }).setView([20.5937, 78.9629], 5);
  L.control.zoom({ position: 'bottomleft' }).addTo(state.map);
  setMapLayer();
}
function setMapLayer() {
  if (state.mapTileLayer) state.map.removeLayer(state.mapTileLayer);
  const urls = {
    light: 'https://{s}.basemaps.cartocdn.com/voyager/{z}/{x}/{y}{r}.png',
    street: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    satellite: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    dark: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
  };
  state.mapTileLayer = L.tileLayer(urls[state.mapLayer] || urls.light, {
    attribution: '&copy; OpenStreetMap & CartoDB',
    maxZoom: 19
  }).addTo(state.map);
}
function toggleMapLayer() {
  const layers = ['light','street','satellite','dark'];
  const idx = layers.indexOf(state.mapLayer);
  state.mapLayer = layers[(idx+1)%layers.length];
  setMapLayer();
  showToast(`Map: ${state.mapLayer}`, 'info');
}
function toggleRouteLines() {
  state.showRoutes = !state.showRoutes;
  state.routeLines.forEach(l => state.showRoutes ? l.addTo(state.map) : state.map.removeLayer(l));
  showToast(state.showRoutes ? 'Routes shown' : 'Routes hidden', 'info');
}
function fitMapBounds() {
  if (state.markers.length) {
    const group = L.featureGroup(state.markers);
    state.map.fitBounds(group.getBounds().pad(0.1));
  }
}
function clearMap() {
  state.markers.forEach(m => state.map.removeLayer(m));
  state.routeLines.forEach(l => state.map.removeLayer(l));
  state.markers = [];
  state.routeLines = [];
}

function addMarker(lat, lon, title, type, popupHtml, color) {
  const iconColor = color || '#667eea';
  const icon = L.divIcon({
    className: 'custom-marker',
    html: `<div style="background:${iconColor};color:#fff;width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px;border:2px solid #fff;box-shadow:0 2px 8px rgba(0,0,0,0.3)">${getTypeEmoji(type)}</div>`,
    iconSize: [28, 28], iconAnchor: [14, 14]
  });
  const marker = L.marker([lat, lon], { icon }).addTo(state.map);
  marker.bindPopup(popupHtml || `<b>${title}</b><br><small>${type}</small>`);
  state.markers.push(marker);
  return marker;
}

function addRouteLine(coords, color, dashed) {
  const line = L.polyline(coords, {
    color: color || '#667eea', weight: 3, opacity: 0.7,
    dashArray: dashed ? '8,8' : null
  });
  if (state.showRoutes) line.addTo(state.map);
  state.routeLines.push(line);
  return line;
}

function getTypeEmoji(type) {
  const map = {temple:'🛕',museum:'🏛️',fort:'🏰',palace:'🏰',beach:'🏖️',park:'🌳',viewpoint:'👀',monument:'🗿',historic:'📜',attraction:'⭐',zoo:'🦁',garden:'🌺',market:'🛍️',restaurant:'🍽️',cafe:'☕',ruins:'🏚️'};
  if (!type) return '📍';
  const t = type.toLowerCase();
  return map[t] || Object.entries(map).find(([k]) => t.includes(k))?.[1] || '📍';
}

// ============================================
// AGENTS
// ============================================
function renderAgentCards() {
  const container = document.getElementById('agentCards');
  container.innerHTML = AGENTS.map(a => `
    <div class="agent-card" id="agent-${a.id}" data-agent="${a.id}">
      <div class="agent-icon">${a.icon}</div>
      <div class="agent-info">
        <div class="agent-name">${a.name}</div>
        <div class="agent-role">${a.role}</div>
      </div>
      <div class="agent-status idle" id="status-${a.id}"></div>
    </div>
  `).join('');
}

function setAgentStatus(agentId, status) {
  const card = document.getElementById(`agent-${agentId}`);
  const dot = document.getElementById(`status-${agentId}`);
  if (card) { card.className = `agent-card ${status}`; }
  if (dot) { dot.className = `agent-status ${status}`; }
}

function addLog(agentId, message) {
  const agent = AGENTS.find(a => a.id === agentId);
  const log = document.getElementById('activityLog');
  const time = new Date().toLocaleTimeString('en-US', {hour:'2-digit',minute:'2-digit',second:'2-digit'});
  log.innerHTML = `<div class="log-entry"><span class="log-time">${time}</span> ${agent?.icon||'🤖'} <strong>${agent?.name||agentId}</strong>: ${message}</div>` + log.innerHTML;
  // Keep only 30 entries
  while (log.children.length > 30) log.removeChild(log.lastChild);
}

function addConvoMessage(agentId, message) {
  const agent = AGENTS.find(a => a.id === agentId);
  const convo = document.getElementById('agentConvo');
  convo.innerHTML += `<div class="convo-msg"><div class="convo-icon">${agent?.icon||'🤖'}</div><div class="convo-text"><strong>${agent?.name||'Agent'}</strong>: ${message}</div></div>`;
  convo.scrollTop = convo.scrollHeight;
}

// ============================================
// TRIP GENERATION (uses backend API → FREE APIs)
// ============================================
async function generateTrip() {
  const destination = document.getElementById('destination').value.trim();
  const origin = document.getElementById('origin').value.trim();
  const duration = parseInt(document.getElementById('duration').value) || 3;
  const budget = parseInt(document.getElementById('budget').value) || 15000;
  const startDate = document.getElementById('startDate').value;

  if (!destination) { showToast('Please enter a destination!', 'error'); return; }
  if (state.generating) return;
  state.generating = true;
  state.currentDest = destination;
  state.currentOrigin = origin;

  // Show loading with agent animation
  showLoading(true);
  document.getElementById('agentConvoPanel').style.display = 'block';
  document.getElementById('agentConvo').innerHTML = '';

  // Simulate agent activation sequence
  const agentSequence = [
    {id:'planner', msg:`Analyzing ${destination}... running MCTS with 30 iterations for optimal route.`, delay:300},
    {id:'weather', msg:`Fetching weather data from OpenMeteo API... applying Naive Bayes classification.`, delay:600},
    {id:'crowd', msg:`Computing crowd heuristics for time-of-day optimization.`, delay:400},
    {id:'budget', msg:`Optimizing ₹${budget.toLocaleString()} budget using MDP reward function.`, delay:500},
    {id:'preference', msg:`Loading Bayesian Beta priors for ${state.persona} persona.`, delay:300},
    {id:'booking', msg:`Preparing multi-platform booking search for ${origin || 'your location'} → ${destination}.`, delay:400},
    {id:'explain', msg:`Generating MDP decision trace and POMDP belief state.`, delay:300},
  ];

  for (const step of agentSequence) {
    setAgentStatus(step.id, 'working');
    addLog(step.id, step.msg);
    addConvoMessage(step.id, step.msg);
    await sleep(step.delay);
  }

  try {
    const resp = await fetch(`${API_BASE}/api/generate-trip`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ destination, origin, duration, budget, persona: state.persona, startDate })
    });
    const data = await resp.json();

    if (!data.success) throw new Error(data.error || 'Failed to generate trip');

    state.itinerary = data.itinerary;
    state.budget = { total: budget, used: data.itinerary.totalCost, breakdown: data.itinerary.budgetBreakdown };

    // Update AI state
    if (data.itinerary.ai) {
      state.bayesian = data.itinerary.ai.bayesian || state.bayesian;
      state.dirichlet = data.itinerary.ai.dirichlet || {};
      state.pomdpBelief = data.itinerary.ai.pomdp_belief || {};
      state.rl.rewards = data.itinerary.ai.rewards || [];
    }

    // Mark all agents completed
    AGENTS.forEach(a => { setAgentStatus(a.id, 'completed'); addLog(a.id, '✅ Task completed'); });
    addConvoMessage('planner', `✅ ${destination} itinerary ready! ${data.itinerary.days_data.length} days, ${data.itinerary.days_data.reduce((s,d)=>s+d.activities.length,0)} activities.`);
    addConvoMessage('explain', `MDP action selected: <strong>${data.mdpAction}</strong> | Reward: <strong>${data.reward?.toFixed(3)}</strong> | Q-table entries: ${data.itinerary.ai?.q_table_size || 0}`);

    // Render everything
    renderItinerary(data.itinerary);
    renderMap(data.itinerary);
    renderWeather(data.itinerary.weather);
    renderBudget(data.itinerary);
    renderBayesian();
    renderDirichlet();
    renderPOMDP();
    renderRLChart();
    renderBudgetChart();
    renderLanguageTips(data.languageTips);
    renderPackingList(data.packingList);
    renderDiscovery(data.itinerary, data.restaurants);
    renderInsights(data.itinerary);
    updateCrowdLevel(data.itinerary);
    renderExplainability(data);
    renderAgentGraph();
    showBookingWizard(data.itinerary);

    // Update atlas
    addToAtlas(destination, data.itinerary);
    
    // Save for comparison
    _addTripToSaved(data.itinerary);
    
    // Render emergency contacts & safety tips
    if (data.emergencyContacts) renderEmergencyContacts(data.emergencyContacts);
    if (data.safetyTips) renderSafetyTips(data.safetyTips);

    document.getElementById('insightsPanel').style.display = 'block';
    showToast(`✅ ${destination} trip generated with ${data.itinerary.days_data.reduce((s,d)=>s+d.activities.length,0)} activities!`, 'success');

    // Auto-scroll to booking wizard with agentic AI feel
    setTimeout(() => {
      const wizard = document.getElementById('agenticWizard');
      if (wizard) {
        wizard.scrollIntoView({ behavior: 'smooth', block: 'center' });
        // Add a pulse animation to draw attention
        wizard.style.animation = 'wizardPulse 0.6s ease-in-out 2';
        setTimeout(() => { wizard.style.animation = ''; }, 1200);
      }
    }, 800);

  } catch(err) {
    console.error('Trip generation error:', err);
    showToast('Error generating trip: ' + err.message, 'error');
    AGENTS.forEach(a => setAgentStatus(a.id, 'idle'));
  } finally {
    state.generating = false;
    showLoading(false);
  }
}

// ============================================
// RENDER ITINERARY
// ============================================
function renderItinerary(itin) {
  const container = document.getElementById('itineraryContainer');
  if (!itin?.days_data?.length) { container.innerHTML = '<div class="empty-state"><div class="emoji">📭</div><p>No itinerary data.</p></div>'; return; }

  container.innerHTML = `<div class="section-title"><i class="fas fa-route"></i> ${itin.destination} — ${itin.days} Day Itinerary (${itin.persona})</div>` +
    itin.days_data.map(day => `
      <div class="day-card" id="day-${day.day}">
        <div class="day-header" onclick="this.parentElement.classList.toggle('collapsed')">
          <div class="day-title">${day.weather?.icon||'☀️'} Day ${day.day} — ${day.city} ${day.date ? `<span class="text-xs text-muted">(${day.date})</span>` : ''}</div>
          <div class="day-meta">
            <span>🌡️ ${day.weather?.temp_max||30}°/${day.weather?.temp_min||22}°</span>
            <span>💰 ₹${day.dayBudget?.toLocaleString()||0}</span>
            <span>📍 ${day.activities?.length||0} places</span>
          </div>
        </div>
        <div class="day-body">
          ${(day.activities||[]).map((act, idx) => renderActivityCard(act, day.day, idx)).join('')}
        </div>
      </div>
    `).join('');

  // Fetch photos asynchronously using wikiTitle for accuracy
  itin.days_data.forEach(day => {
    day.activities?.forEach(async (act, idx) => {
      const photo = act.photo || await fetchPlacePhoto(act.name, act.type, act.wikiTitle);
      const img = document.querySelector(`#act-photo-${day.day}-${idx}`);
      if (img && photo) img.src = photo;
    });
  });
}

function renderActivityCard(act, dayNum, idx) {
  const photo = act.photo || PLACEHOLDER_IMG;
  const crowdColor = act.crowd_level > 70 ? 'var(--danger)' : act.crowd_level > 40 ? 'var(--warning)' : 'var(--success)';
  return `
    <div class="activity-card" data-name="${act.name}" data-lat="${act.lat}" data-lon="${act.lon}">
      <img class="activity-photo" id="act-photo-${dayNum}-${idx}" src="${photo}" alt="${act.name}" onerror="this.src='${PLACEHOLDER_IMG}'" loading="lazy">
      <div class="activity-info">
        <div class="activity-name" onclick="openPlaceModal('${act.name.replace(/'/g,"\\'")}',${act.lat},${act.lon},'${(act.type||'').replace(/'/g,"\\'")}','${(act.description||'').replace(/'/g,"\\'").substring(0,100)}')">${act.name}</div>
        <div class="activity-desc">${act.description || act.type}</div>
        <div class="activity-tags">
          <span class="tag tag-time"><i class="fas fa-clock"></i> ${act.time} · ${act.duration}</span>
          <span class="tag tag-cost"><i class="fas fa-rupee-sign"></i> ₹${act.cost}</span>
          <span class="tag tag-type">${getTypeEmoji(act.type)} ${act.type}</span>
          <span class="tag tag-crowd" style="color:${crowdColor}"><i class="fas fa-users"></i> ${act.crowd_level}%</span>
          ${act.weather_warning ? `<span class="tag tag-weather">${act.weather_warning}</span>` : ''}
        </div>
      </div>
      <div class="activity-rating">
        <select onchange="rateActivity('${act.name.replace(/'/g,"\\'")}',this.value,'${act.type}','${state.currentDest}',${dayNum})" title="Rate this place">
          <option value="">⭐</option><option value="5">⭐⭐⭐⭐⭐</option><option value="4">⭐⭐⭐⭐</option><option value="3">⭐⭐⭐</option><option value="2">⭐⭐</option><option value="1">⭐</option>
        </select>
      </div>
    </div>
  `;
}

// ============================================
// MAP RENDERING
// ============================================
function renderMap(itin) {
  clearMap();
  if (!itin?.days_data?.length) return;

  const colors = ['#667eea','#06b6d4','#10b981','#f59e0b','#ef4444','#8b5cf6','#ec4899'];

  // Add origin marker if available
  if (itin.originCoords?.lat) {
    addMarker(itin.originCoords.lat, itin.originCoords.lon, itin.origin || 'Origin', 'origin',
      `<b>🏠 ${itin.origin || 'Origin'}</b>`, '#ef4444');
  }

  // Add activity markers and route lines per day
  itin.days_data.forEach((day, di) => {
    const dayColor = colors[di % colors.length];
    const dayCoords = [];

    day.activities?.forEach(act => {
      if (!act.lat || !act.lon) return;
      dayCoords.push([act.lat, act.lon]);
      addMarker(act.lat, act.lon, act.name, act.type,
        `<b>${act.name}</b><br><small>Day ${day.day} · ${act.time} · ₹${act.cost}</small><br><small>${act.type}</small>`,
        dayColor);
    });

    // Draw route line for the day
    if (dayCoords.length > 1) {
      addRouteLine(dayCoords, dayColor, false);
    }
  });

  // Draw origin-to-destination line
  if (itin.originCoords?.lat && itin.destCoords?.lat) {
    addRouteLine([[itin.originCoords.lat, itin.originCoords.lon], [itin.destCoords.lat, itin.destCoords.lon]], '#ef4444', true);
  }

  fitMapBounds();
}

// ============================================
// WEATHER
// ============================================
function renderWeather(weather) {
  const container = document.getElementById('weatherCards');
  if (!weather?.length) return;
  container.innerHTML = weather.slice(0, 7).map(w => `
    <div class="weather-card" style="border-bottom:2px solid ${w.risk_level==='high'?'var(--danger)':w.risk_level==='medium'?'var(--warning)':'var(--success)'}">
      <div class="weather-icon">${w.icon}</div>
      <div class="weather-temp">${w.temp_max}°</div>
      <div class="weather-desc">Day ${w.day}<br>${w.temp_min}°/${w.temp_max}°</div>
    </div>
  `).join('');
}

// ============================================
// BUDGET
// ============================================
function renderBudget(itin) {
  const b = itin.budgetBreakdown;
  const total = itin.budget;
  const used = (b.accommodation||0) + (b.food||0) + (b.activities||0) + (b.transport||0) + (b.emergency||0);
  const pct = Math.min(100, Math.round(used/total*100));

  document.getElementById('budgetAmount').textContent = `₹${used.toLocaleString()}`;
  document.getElementById('budgetTotal').textContent = `/ ₹${total.toLocaleString()}`;
  document.getElementById('budgetFill').style.width = `${pct}%`;
  document.getElementById('budgetFill').style.background = pct > 90 ? 'var(--danger)' : pct > 70 ? 'var(--warning)' : 'var(--gradient)';

  document.getElementById('budgetCats').innerHTML = [
    ['🏨 Stay', b.accommodation], ['🍽️ Food', b.food], ['🎯 Activities', b.activities],
    ['🚗 Transport', b.transport], ['🆘 Emergency', b.emergency]
  ].map(([label, val]) => `<div class="budget-cat"><span>${label}</span><span class="fw-600">₹${(val||0).toLocaleString()}</span></div>`).join('');

  state.budget = { total, used, breakdown: b };
}

function renderBudgetChart() {
  const canvas = document.getElementById('budgetChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const b = state.budget.breakdown;
  if (!b || !b.accommodation) return;

  if (state.budgetChart) state.budgetChart.destroy();
  state.budgetChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Accommodation','Food','Activities','Transport','Emergency'],
      datasets: [{
        data: [b.accommodation,b.food,b.activities,b.transport,b.emergency],
        backgroundColor: ['#667eea','#10b981','#f59e0b','#8b5cf6','#ef4444'],
        borderWidth: 0
      }]
    },
    options: { responsive: true, maintainAspectRatio: true, plugins: { legend: { display: false } } }
  });
}

// ============================================
// BAYESIAN / DIRICHLET / POMDP
// ============================================
function renderBayesian() {
  const container = document.getElementById('bayesianBars');
  const cats = state.bayesian;
  container.innerHTML = Object.entries(cats).map(([cat, {a, b}]) => {
    const mean = a / (a + b);
    const pct = Math.round(mean * 100);
    const lo = Math.max(0, mean - 1.96 * Math.sqrt(a*b / ((a+b)**2 * (a+b+1))));
    const hi = Math.min(1, mean + 1.96 * Math.sqrt(a*b / ((a+b)**2 * (a+b+1))));
    return `<div style="margin-bottom:6px">
      <div class="flex-between"><span class="text-xs">${cat.charAt(0).toUpperCase()+cat.slice(1)}</span><span class="text-xs text-muted">${pct}% (α=${a}, β=${b})</span></div>
      <div class="progress-bar"><div class="progress-fill" style="width:${pct}%;background:${pct>60?'var(--success)':pct>30?'var(--warning)':'var(--danger)'}"></div></div>
      <div class="text-xs text-muted">95% CI: [${(lo*100).toFixed(0)}%, ${(hi*100).toFixed(0)}%]</div>
    </div>`;
  }).join('');
}

function renderDirichlet() {
  const panel = document.getElementById('dirichletPanel');
  const d = state.dirichlet;
  if (!d || !Object.keys(d).length) return;
  const total = Object.values(d).reduce((s,v) => s+v, 0);
  panel.innerHTML = Object.entries(d).map(([cat, alpha]) => {
    const pct = Math.round(alpha / total * 100);
    return `<div style="margin-bottom:4px"><div class="flex-between"><span class="text-xs">${cat}</span><span class="text-xs text-muted">${pct}% (α=${alpha.toFixed(1)})</span></div>
    <div class="progress-bar"><div class="progress-fill" style="width:${pct}%;background:var(--accent)"></div></div></div>`;
  }).join('');
}

function renderPOMDP() {
  const panel = document.getElementById('pomdpPanel');
  const b = state.pomdpBelief;
  if (!b || !Object.keys(b).length) return;
  const colors = {excellent:'var(--success)',good:'var(--primary)',average:'var(--warning)',poor:'var(--danger)'};
  panel.innerHTML = Object.entries(b).map(([s, prob]) => {
    const pct = Math.round(prob * 100);
    return `<div style="margin-bottom:4px"><div class="flex-between"><span class="text-xs">${s}</span><span class="text-xs fw-600">${pct}%</span></div>
    <div class="progress-bar"><div class="progress-fill" style="width:${pct}%;background:${colors[s]||'var(--primary)'}"></div></div></div>`;
  }).join('');
}

// ============================================
// RL CHART
// ============================================
function renderRLChart() {
  const canvas = document.getElementById('rlChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const rewards = state.rl.rewards.length ? state.rl.rewards : (state.itinerary?.ai?.rewards || []);
  if (!rewards.length) return;

  if (state.rlChart) state.rlChart.destroy();
  state.rlChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: rewards.map((_, i) => `E${i+1}`),
      datasets: [{
        label: 'Reward',
        data: rewards,
        borderColor: '#667eea',
        backgroundColor: 'rgba(102,126,234,0.1)',
        fill: true,
        tension: 0.4,
        pointRadius: 3,
        borderWidth: 2,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: { y: { grid: { color: 'rgba(255,255,255,0.05)' } }, x: { grid: { display: false } } },
      plugins: { legend: { display: false } }
    }
  });
}

// ============================================
// RATE ACTIVITY (updates Bayesian + POMDP + Q-Learning on backend)
// ============================================
async function rateActivity(actName, rating, category, dest, day) {
  if (!rating) return;
  rating = parseInt(rating);
  try {
    const resp = await fetch(`${API_BASE}/api/rate`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ activity: actName, rating, category: category||'cultural', destination: dest, day })
    });
    const data = await resp.json();
    if (data.success) {
      state.bayesian = data.bayesian || state.bayesian;
      state.dirichlet = data.dirichlet || state.dirichlet;
      state.pomdpBelief = data.pomdpBelief || state.pomdpBelief;
      state.rl.rewards = data.rewards || state.rl.rewards;
      renderBayesian();
      renderDirichlet();
      renderPOMDP();
      renderRLChart();
      addLog('preference', `Rated ${actName}: ${'⭐'.repeat(rating)} → Bayesian updated`);
      showToast(`Rated ${actName}: ${'⭐'.repeat(rating)}`, 'success');
    }
  } catch(e) { console.error('Rating error:', e); }
}

// ============================================
// LANGUAGE TIPS
// ============================================
function renderLanguageTips(tips) {
  const section = document.getElementById('languageTips');
  if (!tips?.phrases?.length) return;
  section.innerHTML = `
    <div class="section-title">🗣️ Language Tips — ${tips.language}</div>
    <div class="lang-grid">
      ${tips.phrases.map(p => `
        <div class="lang-card">
          <div class="lang-phrase">${p.phrase}</div>
          <div class="lang-meaning">${p.meaning}</div>
          <div class="lang-pronunciation">/${p.pronunciation}/</div>
        </div>
      `).join('')}
    </div>
  `;
}

// ============================================
// PACKING LIST (from NOMAD concept)
// ============================================
function renderPackingList(list) {
  state.packingList = list || {};
  const container = document.getElementById('packingList');
  if (!list || !Object.keys(list).length) return;

  const catEmojis = {'Essentials':'📋','Clothing':'👕','Toiletries':'🧴','Tech':'📱','Travel Comfort':'😌','Weather Prep':'☔','Adventure Gear':'🏔️','Luxury':'💎'};
  container.innerHTML = Object.entries(list).map(([cat, items]) => `
    <div class="packing-category">
      <div class="packing-category-title">${catEmojis[cat]||'📦'} ${cat} <span class="text-xs text-muted">(${items.length})</span></div>
      ${items.map((item, i) => {
        const id = `pack-${cat}-${i}`;
        const checked = state.packingChecked[id] || false;
        return `<div class="packing-item ${checked?'checked':''}">
          <input type="checkbox" id="${id}" ${checked?'checked':''} onchange="togglePackItem('${id}')">
          <label for="${id}">${item}</label>
        </div>`;
      }).join('')}
    </div>
  `).join('');
  updatePackingProgress();
}

function togglePackItem(id) {
  state.packingChecked[id] = !state.packingChecked[id];
  localStorage.setItem('sr-packing', JSON.stringify(state.packingChecked));
  const item = document.getElementById(id)?.closest('.packing-item');
  if (item) item.classList.toggle('checked', state.packingChecked[id]);
  updatePackingProgress();
}

function updatePackingProgress() {
  const total = Object.values(state.packingList).flat().length;
  const checked = Object.values(state.packingChecked).filter(Boolean).length;
  const pct = total ? Math.round(checked/total*100) : 0;
  const fill = document.getElementById('packingProgress');
  const count = document.getElementById('packingCount');
  if (fill) fill.style.width = `${pct}%`;
  if (count) count.textContent = `${checked}/${total} packed`;
}

// ============================================
// ATLAS (from NOMAD)
// ============================================
function addToAtlas(destination, itin) {
  const exists = state.atlasTrips.find(t => t.destination.toLowerCase() === destination.toLowerCase());
  if (!exists) {
    state.atlasTrips.push({
      destination, date: new Date().toISOString().split('T')[0],
      lat: itin.destCoords?.lat || 20, lon: itin.destCoords?.lon || 78,
      days: itin.days, budget: itin.budget
    });
    localStorage.setItem('sr-atlas', JSON.stringify(state.atlasTrips));
  }
  updateAtlasStats();
}

function updateAtlasStats() {
  document.getElementById('countriesVisited').textContent = new Set(state.atlasTrips.map(t => t.destination)).size;
  document.getElementById('tripsCount').textContent = state.atlasTrips.length;
  document.getElementById('totalDistance').textContent = Math.round(state.atlasTrips.length * 450);
  document.getElementById('continentsVisited').textContent = Math.min(state.atlasTrips.length, 6);
}

function renderAtlasMap() {
  if (state.atlasMap) { state.atlasMap.remove(); state.atlasMap = null; }
  state.atlasMap = L.map('atlasMap').setView([20, 78], 4);
  const atlasUrl = state.theme === 'dark' ? 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png' : 'https://{s}.basemaps.cartocdn.com/voyager/{z}/{x}/{y}{r}.png';
  L.tileLayer(atlasUrl, { attribution: '&copy; OSM & CartoDB', maxZoom: 19 }).addTo(state.atlasMap);

  state.atlasTrips.forEach(trip => {
    L.circleMarker([trip.lat, trip.lon], {
      radius: 8, fillColor: '#667eea', color: '#fff', weight: 2, fillOpacity: 0.8
    }).addTo(state.atlasMap).bindPopup(`<b>${trip.destination}</b><br>${trip.date}<br>${trip.days} days · ₹${trip.budget?.toLocaleString()}`);
  });

  // Draw lines between trips
  if (state.atlasTrips.length > 1) {
    const coords = state.atlasTrips.map(t => [t.lat, t.lon]);
    L.polyline(coords, { color: '#667eea', weight: 2, opacity: 0.5, dashArray: '5,5' }).addTo(state.atlasMap);
  }
}

// ============================================
// VIEWS
// ============================================
function switchView(view) {
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.toggle('active', t.dataset.view === view));
  document.getElementById('view-planner').style.display = view === 'planner' ? 'grid' : 'none';
  document.getElementById('view-atlas').style.display = view === 'atlas' ? 'block' : 'none';
  document.getElementById('view-packing').style.display = view === 'packing' ? 'block' : 'none';
  const dashEl = document.getElementById('view-dashboard');
  if (dashEl) dashEl.style.display = view === 'dashboard' ? 'block' : 'none';
  document.getElementById('bottomPanels').style.display = view === 'planner' ? 'grid' : 'none';

  if (view === 'atlas') { setTimeout(() => renderAtlasMap(), 100); }
  if (view === 'planner' && state.map) { setTimeout(() => state.map.invalidateSize(), 100); }
  if (view === 'dashboard') { setTimeout(() => renderDashboard(), 100); }
}

// ============================================
// DISCOVERY (Viral/Hidden Gems/Foodie)
// ============================================
function renderDiscovery(itin, restaurants) {
  // Instagram-style trending
  const instaGrid = document.getElementById('instaGrid');
  const activities = itin.days_data.flatMap(d => d.activities).slice(0, 6);
  instaGrid.innerHTML = activities.map(act => `
    <div class="discovery-card" onclick="openPlaceModal('${act.name.replace(/'/g,"\\'")}',${act.lat},${act.lon},'${act.type}','')">
      <img src="${act.photo || PLACEHOLDER_IMG}" alt="${act.name}" onerror="this.src='${PLACEHOLDER_IMG}'" loading="lazy" data-wiki="${act.wikiTitle||act.name}" class="disc-img">
      <div class="discovery-card-body">
        <div class="discovery-card-title">${act.name}</div>
        <div class="discovery-card-meta">📍 ${itin.destination} · ${act.type} · ₹${act.cost}</div>
      </div>
    </div>
  `).join('');
  document.querySelector('#disc-instagram .empty-state')?.remove();
  // Async load discovery photos
  document.querySelectorAll('.disc-img').forEach(async img => {
    const wiki = img.getAttribute('data-wiki');
    if (wiki && img.src.includes('data:image')) {
      const url = await fetchPlacePhoto(wiki, '', wiki);
      if (url && !url.includes('data:image')) img.src = url;
    }
  });

  // YouTube hidden gems
  const ytGrid = document.getElementById('ytGrid');
  const hiddenGems = itin.days_data.flatMap(d => d.activities).filter(a => a.crowd_level < 40).slice(0, 4);
  ytGrid.innerHTML = hiddenGems.map(act => `
    <div class="discovery-card">
      <img src="${act.photo || PLACEHOLDER_IMG}" alt="${act.name}" onerror="this.src='${PLACEHOLDER_IMG}'" loading="lazy">
      <div class="discovery-card-body">
        <div class="discovery-card-title">💎 ${act.name}</div>
        <div class="discovery-card-meta">Low crowd (${act.crowd_level}%) · Hidden Gem</div>
      </div>
    </div>
  `).join('');
  document.querySelector('#disc-youtube .empty-state')?.remove();

  // Foodie spots
  const foodGrid = document.getElementById('foodGrid');
  if (restaurants?.length) {
    foodGrid.innerHTML = restaurants.slice(0, 6).map(r => `
      <div class="discovery-card">
        <div style="height:80px;background:var(--bg-4);display:flex;align-items:center;justify-content:center;font-size:2.5rem">🍽️</div>
        <div class="discovery-card-body">
          <div class="discovery-card-title">${r.name}</div>
          <div class="discovery-card-meta">${r.cuisine} · ⭐ ${r.rating} · ${r.price_range} · ~₹${r.avgCost}</div>
        </div>
      </div>
    `).join('');
    document.querySelector('#disc-foodie .empty-state')?.remove();
  }
}

function switchDiscTab(tab, btn) {
  document.querySelectorAll('.disc-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.disc-content').forEach(c => { c.classList.remove('active'); c.style.display = 'none'; });
  if (btn) btn.classList.add('active');
  const el = document.getElementById(`disc-${tab}`);
  if (el) { el.classList.add('active'); el.style.display = 'block'; }
}

// ============================================
// INSIGHTS & SAFETY
// ============================================
function renderInsights(itin) {
  const container = document.getElementById('insightsContainer');
  const weather = itin.weather || [];
  const rainyDays = weather.filter(w => w.risk_level === 'high').length;
  const totalAct = itin.days_data.reduce((s,d) => s + d.activities.length, 0);
  const avgCrowd = Math.round(itin.days_data.flatMap(d => d.activities).reduce((s,a) => s + (a.crowd_level||50), 0) / totalAct);

  container.innerHTML = `
    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:8px">
      <div class="tag tag-info" style="padding:10px;font-size:0.82rem">📊 ${totalAct} activities across ${itin.days} days</div>
      <div class="tag ${rainyDays?'tag-warning':'tag-success'}" style="padding:10px;font-size:0.82rem">${rainyDays ? `🌧️ ${rainyDays} rainy day(s) — outdoor activities adjusted` : '☀️ Good weather expected!'}</div>
      <div class="tag ${avgCrowd>60?'tag-warning':'tag-success'}" style="padding:10px;font-size:0.82rem">👥 Avg crowd: ${avgCrowd}% — ${avgCrowd>60?'Consider early mornings':'Comfortable levels'}</div>
      <div class="tag tag-info" style="padding:10px;font-size:0.82rem">💰 Budget utilization: ${Math.round(itin.totalCost/itin.budget*100)}%</div>
    </div>
    <div style="margin-top:12px;padding:12px;background:var(--bg-3);border-radius:var(--radius-sm);font-size:0.82rem;color:var(--text-2)">
      <strong>🛡️ Safety Tips:</strong><br>
      • Keep copies of all documents and share itinerary with family<br>
      • Use registered taxis only · Emergency: 112 (India)<br>
      • Stay hydrated, carry water bottle · Apply sunscreen regularly<br>
      • Download offline maps for ${itin.destination} via Google Maps
    </div>
  `;
}

// ============================================
// EXPLAINABILITY & AGENT GRAPH
// ============================================
function renderExplainability(data) {
  const panel = document.getElementById('explainPanel');
  const ai = data.itinerary?.ai;
  if (!ai) return;
  panel.innerHTML = `
    <div class="text-xs" style="line-height:1.6">
      <strong>MDP Action:</strong> ${data.mdpAction}<br>
      <strong>Reward:</strong> R = ${data.reward?.toFixed(4)}<br>
      <strong>Q-Table Size:</strong> ${ai.q_table_size} entries<br>
      <strong>ε-Greedy:</strong> ε = ${ai.epsilon?.toFixed(3)}<br>
      <strong>MCTS:</strong> ${ai.mcts_iterations} iterations<br>
      <strong>Formula:</strong> R = 0.4·rating + 0.3·budget + 0.2·weather − 0.1·crowd
    </div>
  `;
}

function renderAgentGraph() {
  const canvas = document.getElementById('agentGraphCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.parentElement.clientWidth;
  const H = 200;
  canvas.width = W;
  canvas.height = H;
  ctx.clearRect(0, 0, W, H);

  const cx = W / 2, cy = H / 2, radius = 70;
  const positions = AGENTS.map((a, i) => {
    const angle = (i / AGENTS.length) * 2 * Math.PI - Math.PI / 2;
    return { x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle), agent: a };
  });

  // Draw connections
  ctx.strokeStyle = 'rgba(102,126,234,0.2)';
  ctx.lineWidth = 1;
  const connections = [[0,1],[0,2],[0,3],[0,4],[1,3],[2,3],[3,5],[4,0],[5,6],[6,0]];
  connections.forEach(([a,b]) => {
    ctx.beginPath();
    ctx.moveTo(positions[a].x, positions[a].y);
    ctx.lineTo(positions[b].x, positions[b].y);
    ctx.stroke();
  });

  // Draw nodes
  positions.forEach(p => {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 18, 0, 2 * Math.PI);
    ctx.fillStyle = p.agent.color + '30';
    ctx.fill();
    ctx.strokeStyle = p.agent.color;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.font = '14px serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(p.agent.icon, p.x, p.y);
    ctx.font = '8px Inter, sans-serif';
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text-2');
    ctx.fillText(p.agent.name.split(' ')[0], p.x, p.y + 26);
  });
}

function updateCrowdLevel(itin) {
  const acts = itin.days_data.flatMap(d => d.activities);
  const avg = acts.length ? Math.round(acts.reduce((s,a) => s+a.crowd_level, 0) / acts.length) : 0;
  document.getElementById('crowdLabel').textContent = avg > 70 ? `High (${avg}%)` : avg > 40 ? `Medium (${avg}%)` : `Low (${avg}%)`;
  const segments = document.querySelectorAll('#crowdBar .crowd-segment');
  const level = Math.ceil(avg / 20);
  segments.forEach((s, i) => {
    s.style.background = i < level ? (i < 2 ? 'var(--success)' : i < 4 ? 'var(--warning)' : 'var(--danger)') : 'var(--bg-4)';
  });
}

// ============================================
// BOOKING WIZARD
// ============================================
function showBookingWizard(itin) {
  document.getElementById('agenticWizard').style.display = 'block';
  document.querySelector('[data-step="trip_planned"]').classList.add('completed');
  
  // Agentic AI prompt — typing effect
  const dest = itin.destination || 'your destination';
  const actCount = itin.days_data?.reduce((s,d)=>s+d.activities.length,0) || 0;
  const promptText = document.getElementById('agentPromptText');
  const fullText = `🎯 Your ${dest} trip is ready — ${actCount} activities across ${itin.days} days! I've found the best flights, trains, hotels, and cabs. Let me help you book everything seamlessly.`;
  promptText.textContent = '';
  let charIdx = 0;
  const typeInterval = setInterval(() => {
    if (charIdx < fullText.length) {
      promptText.textContent += fullText[charIdx];
      charIdx++;
    } else {
      clearInterval(typeInterval);
    }
  }, 15);
}

async function searchFlights() {
  setWizardStep('flights');
  addLog('booking', 'Searching flights...');
  try {
    const r = await fetch(`${API_BASE}/api/search-flights`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({origin:state.currentOrigin||'Chennai', destination:state.currentDest, date:document.getElementById('startDate').value})
    });
    const d = await r.json();
    if (d.success) renderBookingResults('✈️ Flight Options', d.flights, 'flight');
  } catch(e) { showToast('Flight search failed', 'error'); }
}

async function searchTrains() {
  setWizardStep('trains');
  addLog('booking', 'Searching trains...');
  try {
    const r = await fetch(`${API_BASE}/api/search-trains`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({origin:state.currentOrigin||'Chennai', destination:state.currentDest})
    });
    const d = await r.json();
    if (d.success) renderBookingResults('🚂 Train Options', d.trains, 'train');
  } catch(e) { showToast('Train search failed', 'error'); }
}

async function searchHotels() {
  setWizardStep('hotels');
  addLog('booking', 'Searching hotels...');
  try {
    const r = await fetch(`${API_BASE}/api/search-hotels`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({city:state.currentDest, days:state.itinerary?.days||3, persona:state.persona})
    });
    const d = await r.json();
    if (d.success) renderBookingResults('🏨 Hotel Options', d.hotels, 'hotel');
  } catch(e) { showToast('Hotel search failed', 'error'); }
}

async function searchCabs() {
  setWizardStep('cabs');
  addLog('booking', 'Searching local transport...');
  try {
    const r = await fetch(`${API_BASE}/api/search-cabs`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({city:state.currentDest})
    });
    const d = await r.json();
    if (d.success) renderBookingResults('🚗 Local Transport', d.cabs, 'cab');
  } catch(e) { showToast('Cab search failed', 'error'); }
}

function renderBookingResults(title, results, type) {
  const panel = document.getElementById('bookingResultsPanel');
  const titleEl = document.getElementById('bookingResultsTitle');
  const list = document.getElementById('bookingResultsList');
  panel.style.display = 'block';
  titleEl.innerHTML = `<i class="fas fa-search"></i> ${title}`;

  if (type === 'flight') {
    list.innerHTML = results.map(f => `
      <div class="booking-card" onclick="selectBooking('flights',${JSON.stringify(f).replace(/"/g,'&quot;')},this)">
        <div class="booking-card-title">${f.airline} — ${f.flight_no}</div>
        <div class="booking-card-price">₹${f.price.toLocaleString()}</div>
        <div class="booking-card-meta">${f.departure} → ${f.arrival} · ${f.duration} · ${f.class} · ${f.stops===0?'Non-stop':f.stops+' stop(s)'} · ⭐ ${f.rating}</div>
        <div style="margin-top:6px;display:flex;flex-wrap:wrap;gap:4px">
          ${(f.bookingPlatforms||[{name:'Google Flights',url:f.bookingUrl}]).map(p => `<a href="${p.url}" target="_blank" class="tag tag-info" style="text-decoration:none;cursor:pointer">🔗 ${p.name}</a>`).join('')}
        </div>
      </div>
    `).join('');
  } else if (type === 'train') {
    list.innerHTML = results.map(t => `
      <div class="booking-card" onclick="selectBooking('trains',${JSON.stringify(t).replace(/"/g,'&quot;')},this)">
        <div class="booking-card-title">${t.train_name} — ${t.train_no}</div>
        <div class="booking-card-price">₹${t.price.toLocaleString()}</div>
        <div class="booking-card-meta">${t.departure} · ${t.duration} · Class: ${t.class}</div>
        <div style="margin-top:6px;display:flex;flex-wrap:wrap;gap:4px">
          ${(t.bookingPlatforms||[{name:'IRCTC',url:t.bookingUrl}]).map(p => `<a href="${p.url}" target="_blank" class="tag tag-info" style="text-decoration:none;cursor:pointer">🔗 ${p.name}</a>`).join('')}
        </div>
      </div>
    `).join('');
  } else if (type === 'hotel') {
    list.innerHTML = results.map(h => `
      <div class="booking-card" onclick="selectBooking('hotels',${JSON.stringify(h).replace(/"/g,'&quot;')},this)">
        <div class="booking-card-title">${h.name} ${'⭐'.repeat(h.stars)}</div>
        <div class="booking-card-price">₹${h.price_per_night.toLocaleString()}/night</div>
        <div class="booking-card-meta">Total: ₹${h.total_price.toLocaleString()} · Rating: ${h.rating} · ${h.amenities.join(', ')}</div>
        <div style="margin-top:6px;display:flex;flex-wrap:wrap;gap:4px">
          ${(h.bookingPlatforms||[{name:'Booking.com',url:h.bookingUrl}]).map(p => `<a href="${p.url}" target="_blank" class="tag tag-info" style="text-decoration:none;cursor:pointer">🔗 ${p.name}</a>`).join('')}
        </div>
      </div>
    `).join('');
  } else if (type === 'cab') {
    list.innerHTML = results.map(c => `
      <div class="booking-card" onclick="selectBooking('cabs',${JSON.stringify(c).replace(/"/g,'&quot;')},this)">
        <div class="booking-card-title">${c.provider} — ${c.type}</div>
        <div class="booking-card-price">₹${c.base_fare} base + ₹${c.price_per_km}/km</div>
        <div class="booking-card-meta">${c.estimated_10km ? `Est. 10km ride: ₹${c.estimated_10km}` : ''}</div>
        <div style="margin-top:6px"><a href="${c.bookingUrl}" target="_blank" class="tag tag-info" style="text-decoration:none;cursor:pointer">🔗 Open ${c.provider}</a></div>
      </div>
    `).join('');
  }
}

function selectBooking(type, item, el) {
  state.bookingCart[type] = item;
  document.querySelectorAll(`#bookingResultsList .booking-card`).forEach(c => c.classList.remove('selected'));
  if (el) el.classList.add('selected');
  showToast(`Selected ${type}: ${item.name || item.airline || item.train_name || item.provider}`, 'success');
}

function setWizardStep(step) {
  document.querySelectorAll('.wizard-step').forEach(s => s.classList.remove('active'));
  document.querySelector(`[data-step="${step}"]`)?.classList.add('active');
  document.getElementById('reviewCartPanel').style.display = 'none';
  document.getElementById('paymentPanel').style.display = 'none';
  document.getElementById('confirmationPanel').style.display = 'none';
}

function skipToReview() { showReviewCart(); }

function showReviewCart() {
  setWizardStep('review');
  document.getElementById('bookingResultsPanel').style.display = 'none';
  document.getElementById('reviewCartPanel').style.display = 'block';

  const cart = state.bookingCart;
  let total = 0;
  const items = [];
  if (cart.flights) { items.push({label:`✈️ ${cart.flights.airline} ${cart.flights.flight_no}`, price:cart.flights.price}); total += cart.flights.price; }
  if (cart.trains) { items.push({label:`🚂 ${cart.trains.train_name}`, price:cart.trains.price}); total += cart.trains.price; }
  if (cart.hotels) { items.push({label:`🏨 ${cart.hotels.name}`, price:cart.hotels.total_price}); total += cart.hotels.total_price; }
  if (cart.cabs) { items.push({label:`🚗 ${cart.cabs.provider} ${cart.cabs.type}`, price:cart.cabs.base_fare}); total += cart.cabs.base_fare; }

  document.getElementById('cartSummary').innerHTML = items.length ?
    items.map(i => `<div class="cart-item"><span>${i.label}</span><span class="fw-600">₹${i.price.toLocaleString()}</span></div>`).join('') :
    '<div class="text-sm text-muted text-center">No items selected. Use the buttons above to search & select.</div>';
  document.getElementById('cartTotal').textContent = `Total: ₹${total.toLocaleString()}`;
}

function editSelections() {
  document.getElementById('reviewCartPanel').style.display = 'none';
  document.getElementById('bookingResultsPanel').style.display = 'block';
}

function proceedToPayment() {
  setWizardStep('payment');
  document.getElementById('reviewCartPanel').style.display = 'none';
  document.getElementById('paymentPanel').style.display = 'block';
  let total = 0;
  Object.values(state.bookingCart).forEach(item => { if (item) total += item.price || item.total_price || item.base_fare || 0; });
  document.getElementById('paymentTotal').textContent = `Total: ₹${total.toLocaleString()}`;
}

function selectPayment(method, el) {
  document.querySelectorAll('.payment-method').forEach(m => m.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('paymentFormFields').style.display = method === 'card' ? 'block' : 'none';
}

function processPayment() {
  setWizardStep('confirmed');
  document.getElementById('paymentPanel').style.display = 'none';
  document.getElementById('confirmationPanel').style.display = 'block';

  // Save to history
  let total = 0;
  Object.values(state.bookingCart).forEach(item => { if (item) total += item.price || item.total_price || item.base_fare || 0; });
  const booking = { id: Date.now(), destination: state.currentDest, date: new Date().toISOString(), total, items: {...state.bookingCart} };
  state.bookingHistory.push(booking);
  localStorage.setItem('sr-history', JSON.stringify(state.bookingHistory));

  document.getElementById('confirmationMsg').textContent = `All bookings for ${state.currentDest} confirmed! Total: ₹${total.toLocaleString()}`;
  showToast('✅ Booking confirmed!', 'success');
  addLog('booking', `✅ Booking confirmed — ₹${total.toLocaleString()}`);
}

// ============================================
// BOOKING HISTORY
// ============================================
function toggleHistory() {
  const overlay = document.getElementById('historyOverlay');
  const sidebar = document.getElementById('historySidebar');
  const isOpen = sidebar.classList.contains('open');
  overlay.classList.toggle('open', !isOpen);
  sidebar.classList.toggle('open', !isOpen);
  if (!isOpen) renderHistory();
}

function renderHistory() {
  const list = document.getElementById('historyList');
  const totalBookings = state.bookingHistory.length;
  const totalSpent = state.bookingHistory.reduce((s,b) => s + b.total, 0);
  document.getElementById('histTotal').textContent = totalBookings;
  document.getElementById('histSpent').textContent = `₹${totalSpent.toLocaleString()}`;

  if (!totalBookings) { list.innerHTML = '<div class="empty-state"><div class="emoji">📋</div><p>No bookings yet.</p></div>'; return; }
  list.innerHTML = state.bookingHistory.map(b => `
    <div class="history-item">
      <div class="fw-600">📍 ${b.destination}</div>
      <div class="text-xs text-muted">${new Date(b.date).toLocaleDateString()} · ₹${b.total.toLocaleString()}</div>
    </div>
  `).reverse().join('');
}

// ============================================
// MODALS
// ============================================
function openPlaceModal(name, lat, lon, type, desc) {
  document.getElementById('mediaModal').classList.add('active');
  document.getElementById('modalTitle').textContent = name;
  document.getElementById('modalInfo').innerHTML = `<p>${desc || type || 'Tourist attraction'}</p><p>📍 Location: ${lat.toFixed(4)}, ${lon.toFixed(4)}</p>`;
  document.getElementById('modalLinks').innerHTML = `
    <a href="https://www.google.com/maps?q=${lat},${lon}" target="_blank">📍 Open in Google Maps</a>
    <a href="https://en.wikipedia.org/wiki/${encodeURIComponent(name)}" target="_blank">📖 Wikipedia</a>
    <a href="https://www.tripadvisor.com/Search?q=${encodeURIComponent(name)}" target="_blank">⭐ TripAdvisor</a>
  `;
  // Map embed
  const mapDiv = document.getElementById('modalMapEmbed');
  mapDiv.innerHTML = '';
  const miniMap = L.map(mapDiv).setView([lat, lon], 15);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(miniMap);
  L.marker([lat, lon]).addTo(miniMap);
  setTimeout(() => miniMap.invalidateSize(), 100);

  // Photos
  fetchPlacePhoto(name, type, name).then(url => {
    document.getElementById('modalPhotos').innerHTML = url && !url.includes('data:image') ? `<img src="${url}" alt="${name}" style="max-width:100%;border-radius:8px">` : '<p class="text-sm text-muted">Loading photo...</p>';
  });
}

function closeMediaModal() { document.getElementById('mediaModal').classList.remove('active'); }
function closeModal(id) { document.getElementById(id).classList.remove('active'); }

function emergencyReplan() { document.getElementById('replanModal').classList.add('active'); }
function openRecommendModal() { document.getElementById('recommendModal').classList.add('active'); }
function openHalfDayModal() { document.getElementById('halfDayModal').classList.add('active'); }

function updateReplanFields() {
  const reason = document.getElementById('replanReason').value;
  document.getElementById('delayFields').style.display = reason === 'delay' ? 'block' : 'none';
  document.getElementById('weatherFields').style.display = reason === 'weather' ? 'block' : 'none';
  document.getElementById('crowdFields').style.display = reason === 'crowd' ? 'block' : 'none';
}

async function doReplan() {
  if (!state.itinerary) { showToast('Generate a trip first!', 'error'); return; }
  const reason = document.getElementById('replanReason').value;
  const day = parseInt(document.getElementById('replanDay').value) || 1;
  showLoading(true);
  try {
    const r = await fetch(`${API_BASE}/api/replan`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ itinerary: state.itinerary, reason, day, delayHours: parseInt(document.getElementById('delayHours').value)||4 })
    });
    const d = await r.json();
    if (d.success) {
      state.itinerary = d.itinerary;
      renderItinerary(d.itinerary);
      renderMap(d.itinerary);
      showToast(`✅ Day ${day} replanned for ${reason}!`, 'success');
      addLog('planner', `Emergency replan: ${reason} on Day ${day}`);
    }
  } catch(e) { showToast('Replan failed', 'error'); }
  showLoading(false);
  closeModal('replanModal');
}

// ============================================
// RECOMMENDATIONS
// ============================================
async function getRecommendations() {
  const budget = parseInt(document.getElementById('recBudget').value) || 20000;
  const duration = parseInt(document.getElementById('recDuration').value) || 3;
  const prefs = [...document.querySelectorAll('.rec-pref:checked')].map(c => c.value);
  const location = document.getElementById('recLocation').value;

  try {
    const r = await fetch(`${API_BASE}/api/recommendations`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({budget, duration, preferences: prefs, currentLocation: location})
    });
    const d = await r.json();
    if (d.success && d.destinations?.length) {
      document.getElementById('recommendResults').innerHTML = d.destinations.map(dest => `
        <div class="rec-card" onclick="selectRecommendation('${dest.name}')">
          <div class="rec-match">${Math.round(dest.matchScore)}% match</div>
          <div class="rec-name">${dest.name}</div>
          <div class="rec-state">${dest.state}</div>
          <div class="rec-cost">~₹${dest.estimatedCost.toLocaleString()} for ${duration} days</div>
          <div class="rec-tags">${dest.tags.map(t => `<span class="tag tag-type">${t}</span>`).join('')}</div>
          <div class="rec-highlights">✨ ${dest.highlights.join(' · ')}</div>
        </div>
      `).join('');
    } else {
      document.getElementById('recommendResults').innerHTML = '<p class="text-sm text-muted">No matching destinations found. Try adjusting budget or preferences.</p>';
    }
  } catch(e) { showToast('Recommendation error', 'error'); }
}

function selectRecommendation(name) {
  document.getElementById('destination').value = name;
  closeModal('recommendModal');
  showToast(`Selected: ${name}. Click "Generate AI Trip" to plan!`, 'info');
}

// ============================================
// HALF-DAY PLANNER
// ============================================
async function planHalfDay() {
  const location = document.getElementById('hdLocation').value.trim();
  if (!location) { showToast('Enter a location!', 'error'); return; }
  const hours = parseInt(document.getElementById('hdHours').value) || 5;
  const budget = parseInt(document.getElementById('hdBudget').value) || 3000;

  document.getElementById('destination').value = location;
  document.getElementById('duration').value = 1;
  document.getElementById('budget').value = budget;
  closeModal('halfDayModal');
  generateTrip();
}

// ============================================
// NEARBY PLACES
// ============================================
async function findNearbyPlaces() {
  if (!navigator.geolocation) { showToast('Geolocation not supported', 'error'); return; }
  showToast('Finding nearby places...', 'info');
  navigator.geolocation.getCurrentPosition(async (pos) => {
    try {
      const r = await fetch(`${API_BASE}/api/nearby?lat=${pos.coords.latitude}&lon=${pos.coords.longitude}&radius=3000`);
      const d = await r.json();
      if (d.success && d.places?.length) {
        const panel = document.getElementById('nearbyPanel');
        panel.style.display = 'block';
        document.getElementById('nearbyContainer').innerHTML = d.places.map(p => `
          <div class="activity-card" style="margin-bottom:6px;cursor:pointer" onclick="openPlaceModal('${p.name.replace(/'/g,"\\'")}',${p.lat},${p.lon},'${p.type}','')">
            <div style="font-size:1.5rem;width:40px;text-align:center">${getTypeEmoji(p.type)}</div>
            <div class="activity-info"><div class="activity-name">${p.name}</div><div class="activity-desc">${p.type}</div></div>
          </div>
        `).join('');
        showToast(`Found ${d.places.length} nearby places!`, 'success');
      } else { showToast('No places found nearby', 'info'); }
    } catch(e) { showToast('Nearby search failed', 'error'); }
  }, () => showToast('Location access denied', 'error'));
}

// ============================================
// CHATBOT
// ============================================
function toggleChatbot() {
  state.chatOpen = !state.chatOpen;
  document.getElementById('chatbotWindow').classList.toggle('open', state.chatOpen);
}

async function sendChat() {
  const input = document.getElementById('chatInput');
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';

  const messages = document.getElementById('chatMessages');
  messages.innerHTML += `<div class="chat-msg user"><div class="chat-msg-bubble">${msg}</div></div>`;
  messages.scrollTop = messages.scrollHeight;

  try {
    const r = await fetch(`${API_BASE}/api/chat`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({message: msg, context:{destination:state.currentDest}})
    });
    const d = await r.json();
    messages.innerHTML += `<div class="chat-msg bot"><div class="chat-msg-avatar">🧠</div><div class="chat-msg-bubble">${d.response?.replace(/\n/g,'<br>')}</div></div>`;
  } catch(e) {
    messages.innerHTML += `<div class="chat-msg bot"><div class="chat-msg-avatar">🧠</div><div class="chat-msg-bubble">Sorry, I'm having trouble connecting. Please try again!</div></div>`;
  }
  messages.scrollTop = messages.scrollHeight;
}

function sendSuggestion(text) {
  document.getElementById('chatInput').value = text;
  sendChat();
}

// ============================================
// CURRENCY CONVERTER (FREE API)
// ============================================
async function convertCurrency() {
  const amount = parseFloat(document.getElementById('currAmount').value) || 0;
  const from = document.getElementById('currFrom').value;
  const to = document.getElementById('currTo').value;
  // Using exchangerate.host (free, no API key)
  try {
    const r = await fetch(`https://api.exchangerate.host/convert?from=${from}&to=${to}&amount=${amount}`);
    const d = await r.json();
    if (d.result) {
      document.getElementById('currResult').textContent = `${d.result.toFixed(2)} ${to}`;
    } else {
      // Fallback: approximate rates
      const rates = {INR:1,USD:0.012,EUR:0.011,GBP:0.0095,JPY:1.8,THB:0.41};
      const inINR = amount / (rates[from]||1);
      const result = inINR * (rates[to]||1);
      document.getElementById('currResult').textContent = `~${result.toFixed(2)} ${to}`;
    }
  } catch(e) {
    const rates = {INR:1,USD:0.012,EUR:0.011,GBP:0.0095,JPY:1.8,THB:0.41};
    const inINR = amount / (rates[from]||1);
    const result = inINR * (rates[to]||1);
    document.getElementById('currResult').textContent = `~${result.toFixed(2)} ${to}`;
  }
}

// ============================================
// WORLD CLOCK
// ============================================
function updateClocks() {
  const now = new Date();
  document.getElementById('tzLocal').textContent = now.toLocaleTimeString('en-US', {hour:'2-digit',minute:'2-digit'});
  // Estimate destination timezone (simplified)
  const destName = document.getElementById('tzDestName');
  const destTime = document.getElementById('tzDest');
  if (state.currentDest) {
    destName.textContent = state.currentDest;
    destTime.textContent = now.toLocaleTimeString('en-US', {hour:'2-digit',minute:'2-digit',timeZone:'Asia/Kolkata'});
  }
}

// ============================================
// UTILITY FUNCTIONS
// ============================================
function showLoading(show) {
  const overlay = document.getElementById('loadingOverlay');
  overlay.classList.toggle('active', show);
  if (show) {
    document.getElementById('loadingAgents').innerHTML = AGENTS.map((a,i) =>
      `<div class="loading-agent" style="animation-delay:${i*0.15}s">${a.icon}</div>`
    ).join('');
  }
}

function showToast(message, type='info') {
  const container = document.getElementById('toastContainer');
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function selectPersona(persona) {
  state.persona = persona;
  document.querySelectorAll('.persona-card').forEach(c => c.classList.toggle('active', c.dataset.persona === persona));
  showToast(`Persona: ${persona}`, 'info');
}

async function checkBackend() {
  try {
    const r = await fetch(`${API_BASE}/api/health`);
    const d = await r.json();
    document.getElementById('backendStatus').innerHTML = `<span style="color:var(--success)">✅ Connected — ${d.agents} Agents · ${d.engine}</span>`;
  } catch(e) {
    document.getElementById('backendStatus').innerHTML = `<span style="color:var(--danger)">❌ Backend offline</span>`;
  }
}

function detectUserLocation() {
  if (!navigator.geolocation) { showToast('Geolocation not supported', 'error'); return; }
  navigator.geolocation.getCurrentPosition(async (pos) => {
    try {
      const r = await fetch(`https://nominatim.openstreetmap.org/reverse?lat=${pos.coords.latitude}&lon=${pos.coords.longitude}&format=json`, {headers:{'User-Agent':'SmartRouteSRMIST'}});
      const d = await r.json();
      const city = d.address?.city || d.address?.town || d.address?.village || d.display_name?.split(',')[0] || '';
      document.getElementById('origin').value = city;
      showToast(`Location: ${city}`, 'success');
    } catch(e) { showToast('Could not detect location', 'error'); }
  }, () => showToast('Location denied', 'error'));
}

function startVoice() {
  if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
    showToast('Voice input not supported in this browser', 'error'); return;
  }
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const recognition = new SpeechRecognition();
  recognition.lang = 'en-IN';
  recognition.continuous = false;
  recognition.onresult = (e) => {
    const text = e.results[0][0].transcript;
    document.getElementById('destination').value = text;
    showToast(`Voice: "${text}"`, 'success');
  };
  recognition.onerror = () => showToast('Voice error', 'error');
  recognition.start();
  showToast('🎤 Listening...', 'info');
}

function exportPDF() {
  if (!state.itinerary) { showToast('Generate a trip first!', 'error'); return; }
  // Generate printable version
  const printWin = window.open('', '_blank');
  const itin = state.itinerary;
  printWin.document.write(`
    <html><head><title>SmartRoute SRMIST — ${itin.destination} Trip</title>
    <style>body{font-family:Arial,sans-serif;padding:20px;max-width:800px;margin:0 auto}h1{color:#667eea}h2{color:#333;border-bottom:2px solid #667eea;padding-bottom:5px}.activity{padding:8px;margin:4px 0;background:#f5f5f5;border-radius:8px}.tag{display:inline-block;padding:2px 8px;border-radius:12px;font-size:12px;margin:2px;background:#e5e7eb}</style>
    </head><body>
    <h1>🧠 SmartRoute SRMIST — ${itin.destination}</h1>
    <p><strong>Origin:</strong> ${itin.origin} · <strong>Duration:</strong> ${itin.days} days · <strong>Budget:</strong> ₹${itin.budget.toLocaleString()} · <strong>Persona:</strong> ${itin.persona}</p>
    <p><strong>AI Algorithms:</strong> MCTS (${itin.ai?.mcts_iterations} iterations) · Q-Learning (ε=${itin.ai?.epsilon?.toFixed(3)}) · Bayesian Beta · Naive Bayes · POMDP</p>
    <hr>
    ${itin.days_data.map(day => `
      <h2>${day.weather?.icon||''} Day ${day.day} — ${day.city} ${day.date||''}</h2>
      <p>🌡️ ${day.weather?.temp_min||''}°–${day.weather?.temp_max||''}° · 💰 ₹${day.dayBudget?.toLocaleString()||0}</p>
      ${day.activities.map(a => `<div class="activity">
        <strong>${a.name}</strong> (${a.type})<br>
        <span class="tag">🕐 ${a.time} · ${a.duration}</span>
        <span class="tag">💰 ₹${a.cost}</span>
        <span class="tag">👥 Crowd: ${a.crowd_level}%</span>
        ${a.weather_warning?`<span class="tag" style="background:#fee2e2">${a.weather_warning}</span>`:''}
        <br><small>${a.description||''}</small>
      </div>`).join('')}
    `).join('')}
    <hr><p style="text-align:center;color:#999">Generated by SmartRoute SRMIST — Agentic AI Travel Planner</p>
    </body></html>
  `);
  printWin.document.close();
  printWin.print();
}

function shareTrip() {
  if (!state.itinerary) { showToast('Generate a trip first!', 'error'); return; }
  const text = `🧠 SmartRoute SRMIST Trip: ${state.itinerary.destination} (${state.itinerary.days} days, ₹${state.itinerary.budget.toLocaleString()}) - Planned by 7 AI Agents!`;
  if (navigator.share) {
    navigator.share({title:'SmartRoute SRMIST Trip', text, url:window.location.href});
  } else {
    navigator.clipboard?.writeText(text);
    showToast('Trip details copied to clipboard!', 'success');
  }
}

// ============================================
// MULTI-CITY TRIP (from TripSage concept)
// ============================================
function openMultiCityModal() { document.getElementById('multiCityModal').classList.add('active'); }
function addMCCity() {
  const list = document.getElementById('mcCitiesList');
  const idx = list.children.length;
  const row = document.createElement('div');
  row.className = 'mc-city-row';
  row.dataset.idx = idx;
  row.innerHTML = `<input type="text" class="form-input mc-city" placeholder="City ${idx+1}" style="flex:2"><input type="number" class="form-input mc-days" value="2" min="1" max="14" style="width:70px" placeholder="Days"><button class="btn-icon" onclick="removeMCCity(this)" title="Remove"><i class="fas fa-times" style="color:var(--danger)"></i></button>`;
  list.appendChild(row);
}
function removeMCCity(btn) {
  const row = btn.closest('.mc-city-row');
  if (document.querySelectorAll('.mc-city-row').length > 1) row.remove();
  else showToast('Need at least one city', 'error');
}

async function generateMultiCityTrip() {
  const cities = [...document.querySelectorAll('.mc-city')].map(i => i.value.trim()).filter(Boolean);
  const daysPerCity = [...document.querySelectorAll('.mc-days')].map(i => parseInt(i.value) || 2);
  const budget = parseInt(document.getElementById('mcBudget').value) || 30000;
  const origin = document.getElementById('mcOrigin').value.trim() || state.currentOrigin || 'Chennai';
  
  if (!cities.length) { showToast('Add at least one city!', 'error'); return; }
  
  showLoading(true);
  document.getElementById('multiCityResults').innerHTML = '<div class="text-sm text-muted">Planning multi-city route...</div>';
  
  try {
    const r = await fetch(`${API_BASE}/api/generate-multi-city`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({cities, daysPerCity: daysPerCity.slice(0,cities.length), budget, persona:state.persona, origin})
    });
    const d = await r.json();
    if (d.success) {
      // Display multi-city results
      document.getElementById('multiCityResults').innerHTML = `
        <div class="text-sm" style="color:var(--success);margin-bottom:8px">✅ Multi-city trip planned: ${cities.join(' → ')}</div>
        ${(d.cityItineraries||[]).map((ci, idx) => `
          <div class="rec-card" onclick="loadCityItinerary(${idx})" style="cursor:pointer">
            <div class="rec-match">City ${idx+1}</div>
            <div class="rec-name">${ci.destination}</div>
            <div class="rec-cost">${ci.days} days · ₹${ci.totalCost?.toLocaleString()||0}</div>
            <div class="rec-highlights">${ci.days_data?.reduce((s,d)=>s+d.activities.length,0)||0} activities</div>
          </div>
        `).join('')}
      `;
      // Store for loading
      state.multiCityData = d;
      showToast(`✅ Multi-city trip: ${cities.join(' → ')}`, 'success');
    } else { showToast(d.error || 'Multi-city generation failed', 'error'); }
  } catch(e) { showToast('Multi-city trip failed', 'error'); }
  showLoading(false);
}

function loadCityItinerary(idx) {
  if (!state.multiCityData?.cityItineraries?.[idx]) return;
  const ci = state.multiCityData.cityItineraries[idx];
  state.itinerary = ci;
  state.currentDest = ci.destination;
  state.budget = { total: ci.budget, used: ci.totalCost, breakdown: ci.budgetBreakdown };
  
  renderItinerary(ci);
  renderMap(ci);
  renderWeather(ci.weather);
  renderBudget(ci);
  if (ci.languageTips) renderLanguageTips(ci.languageTips);
  
  closeModal('multiCityModal');
  switchView('planner');
  showToast(`Loaded ${ci.destination} itinerary`, 'info');
}

// ============================================
// TRIP COMPARISON (from CrewAI/TripSage concept)
// ============================================
function openCompareModal() {
  document.getElementById('compareModal').classList.add('active');
  renderCompareGrid();
}

function renderCompareGrid() {
  const trips = state.savedTrips || [];
  const grid = document.getElementById('compareGrid');
  
  if (!trips.length) {
    grid.innerHTML = '<div class="empty-state"><div class="emoji">📊</div><p>No trips to compare. Generate trips first!</p></div>';
    return;
  }
  
  grid.innerHTML = `
    <div style="overflow-x:auto">
      <table style="width:100%;border-collapse:collapse;font-size:0.82rem">
        <thead>
          <tr style="border-bottom:2px solid var(--border)">
            <th style="padding:8px;text-align:left;color:var(--text-3)">Metric</th>
            ${trips.map(t => `<th style="padding:8px;text-align:center;color:var(--primary)">${t.destination}</th>`).join('')}
          </tr>
        </thead>
        <tbody>
          <tr><td style="padding:6px">📅 Days</td>${trips.map(t => `<td style="text-align:center;font-weight:600">${t.days}</td>`).join('')}</tr>
          <tr style="background:var(--bg-3)"><td style="padding:6px">💰 Budget</td>${trips.map(t => `<td style="text-align:center">₹${t.budget?.toLocaleString()}</td>`).join('')}</tr>
          <tr><td style="padding:6px">💸 Total Cost</td>${trips.map(t => `<td style="text-align:center">₹${t.totalCost?.toLocaleString()}</td>`).join('')}</tr>
          <tr style="background:var(--bg-3)"><td style="padding:6px">📊 Utilization</td>${trips.map(t => `<td style="text-align:center;color:${(t.totalCost/t.budget)>0.9?'var(--danger)':'var(--success)'}">${Math.round(t.totalCost/t.budget*100)}%</td>`).join('')}</tr>
          <tr><td style="padding:6px">📍 Activities</td>${trips.map(t => `<td style="text-align:center">${t.days_data?.reduce((s,d)=>s+d.activities.length,0)||0}</td>`).join('')}</tr>
          <tr style="background:var(--bg-3)"><td style="padding:6px">🌧️ Rainy Days</td>${trips.map(t => `<td style="text-align:center">${(t.weather||[]).filter(w=>w.risk_level==='high').length}</td>`).join('')}</tr>
          <tr><td style="padding:6px">👥 Avg Crowd</td>${trips.map(t => {
            const acts = t.days_data?.flatMap(d=>d.activities)||[];
            const avg = acts.length ? Math.round(acts.reduce((s,a)=>s+a.crowd_level,0)/acts.length) : 0;
            return `<td style="text-align:center;color:${avg>60?'var(--danger)':avg>40?'var(--warning)':'var(--success)'};">${avg}%</td>`;
          }).join('')}</tr>
        </tbody>
      </table>
    </div>
    <div style="margin-top:12px;text-align:center">
      <button class="btn btn-sm" onclick="clearCompareTrips()" style="color:var(--danger)"><i class="fas fa-trash"></i> Clear All</button>
    </div>
  `;
}

function clearCompareTrips() {
  state.savedTrips = [];
  localStorage.removeItem('sr-saved-trips');
  renderCompareGrid();
  showToast('Comparison cleared', 'info');
}

// ============================================
// EMERGENCY CONTACTS
// ============================================
function renderEmergencyContacts(contacts) {
  const panel = document.getElementById('emergencyContacts');
  if (!contacts) return;
  const icons = {police:'🚔',ambulance:'🚑',fire:'🚒',women_helpline:'👩',tourist_helpline:'🏛️',disaster_mgmt:'⚠️',universal:'🚨',roadside_assistance:'🚗',local_police:'🏪',hospital:'🏥',embassy:'🏳️',tourist_office:'📍'};
  panel.innerHTML = Object.entries(contacts).map(([key, val]) => {
    if (!val) return '';
    const label = key.replace(/_/g,' ').replace(/\b\w/g, c => c.toUpperCase());
    return `<div class="emg-row"><span>${icons[key]||'📞'} ${label}</span><a href="tel:${val}" class="emg-num">${val}</a></div>`;
  }).filter(Boolean).join('');
}

// ============================================
// SAFETY TIPS
// ============================================
function renderSafetyTips(tips) {
  if (!tips?.length) return;
  const container = document.getElementById('insightsContainer');
  if (!container) return;
  // Append safety tips to insights
  const existing = container.innerHTML;
  container.innerHTML = existing + `
    <div style="margin-top:12px;padding:12px;background:var(--bg-3);border-radius:var(--radius-sm);font-size:0.82rem;color:var(--text-2)">
      <strong>🛡️ AI Safety Tips (${tips.length}):</strong><br>
      ${tips.slice(0, 8).map(t => `• ${t}`).join('<br>')}
      ${tips.length > 8 ? `<br><span class="text-xs text-muted">+ ${tips.length - 8} more tips</span>` : ''}
    </div>
  `;
}

// ============================================
// TRIP JOURNAL (from NOMAD notes concept)
// ============================================
function saveJournalEntry() {
  const textarea = document.getElementById('journalEntry');
  const text = textarea.value.trim();
  if (!text) { showToast('Write something first!', 'error'); return; }
  
  const entry = {
    id: Date.now(),
    text,
    date: new Date().toLocaleString(),
    destination: state.currentDest || 'General',
  };
  
  state.journalEntries = state.journalEntries || [];
  state.journalEntries.push(entry);
  localStorage.setItem('sr-journal', JSON.stringify(state.journalEntries));
  textarea.value = '';
  renderJournalEntries();
  showToast('Journal entry saved! 📝', 'success');
}

function renderJournalEntries() {
  const container = document.getElementById('journalEntries');
  const entries = state.journalEntries || [];
  if (!entries.length) { container.innerHTML = '<div class="text-xs text-muted text-center">No entries yet</div>'; return; }
  
  container.innerHTML = entries.slice().reverse().map(e => `
    <div class="journal-entry-card">
      <div class="flex-between"><span class="text-xs fw-600">${e.destination}</span><span class="text-xs text-muted">${e.date}</span></div>
      <div class="text-sm" style="margin-top:4px;color:var(--text-2)">${e.text}</div>
      <button class="btn-icon" style="position:absolute;top:4px;right:4px;width:20px;height:20px;font-size:0.6rem" onclick="deleteJournalEntry(${e.id})"><i class="fas fa-times"></i></button>
    </div>
  `).join('');
}

function deleteJournalEntry(id) {
  state.journalEntries = (state.journalEntries || []).filter(e => e.id !== id);
  localStorage.setItem('sr-journal', JSON.stringify(state.journalEntries));
  renderJournalEntries();
}

// ============================================
// DASHBOARD VIEW (from NOMAD dashboard concept)
// ============================================
function renderDashboard() {
  const trips = state.savedTrips || [];
  const history = state.bookingHistory || [];
  const rewards = state.rl.rewards || [];
  
  document.getElementById('dashTrips').textContent = trips.length;
  document.getElementById('dashBudget').textContent = `₹${trips.reduce((s,t) => s + (t.budget||0), 0).toLocaleString()}`;
  document.getElementById('dashPlaces').textContent = trips.reduce((s,t) => s + (t.days_data?.reduce((s2,d) => s2 + d.activities.length, 0)||0), 0);
  document.getElementById('dashAIActions').textContent = Object.keys(aiState?.qTable||state.rl?.rewards||{}).length || rewards.length;
  document.getElementById('dashRatings').textContent = Object.values(state.bayesian).reduce((s,b) => s + (b.a||0) + (b.b||0), 0) - 14; // minus initial values
  document.getElementById('dashAvgReward').textContent = rewards.length ? (rewards.reduce((s,r) => s+r, 0) / rewards.length).toFixed(3) : '0';
  
  // Render dashboard charts
  renderDashboardCharts();
  
  // Recent activity
  const recent = document.getElementById('dashRecentActivity');
  if (trips.length || history.length) {
    const items = [
      ...trips.map(t => ({time: t._savedAt || new Date().toISOString(), text: `🗺️ Planned: ${t.destination} (${t.days} days)`})),
      ...history.map(b => ({time: b.date, text: `🎫 Booked: ${b.destination} (₹${b.total?.toLocaleString()})`})),
    ].sort((a,b) => new Date(b.time).getTime() - new Date(a.time).getTime()).slice(0,10);
    recent.innerHTML = items.map(i => `<div style="padding:6px 0;border-bottom:1px solid var(--border);font-size:0.82rem"><span class="text-xs text-muted">${new Date(i.time).toLocaleDateString()}</span> ${i.text}</div>`).join('');
  }
}

function renderDashboardCharts() {
  // RL Chart
  const rlCanvas = document.getElementById('dashRLChart');
  if (rlCanvas && state.rl.rewards?.length) {
    const ctx = rlCanvas.getContext('2d');
    if (state._dashRLChart) state._dashRLChart.destroy();
    state._dashRLChart = new Chart(ctx, {
      type: 'line',
      data: { labels: state.rl.rewards.map((_,i) => `E${i+1}`), datasets: [{label:'Reward',data:state.rl.rewards,borderColor:'#667eea',backgroundColor:'rgba(102,126,234,0.1)',fill:true,tension:0.4}] },
      options: { responsive:true, maintainAspectRatio:false, plugins:{legend:{display:false}} }
    });
  }
  
  // Preference Chart
  const prefCanvas = document.getElementById('dashPrefChart');
  if (prefCanvas && state.bayesian) {
    const ctx = prefCanvas.getContext('2d');
    if (state._dashPrefChart) state._dashPrefChart.destroy();
    const cats = Object.entries(state.bayesian);
    state._dashPrefChart = new Chart(ctx, {
      type: 'radar',
      data: {
        labels: cats.map(([k]) => k.charAt(0).toUpperCase()+k.slice(1)),
        datasets: [{label:'Preference',data:cats.map(([_,v]) => v.a/(v.a+v.b)*100),borderColor:'#8b5cf6',backgroundColor:'rgba(139,92,246,0.15)',fill:true}]
      },
      options: { responsive:true, maintainAspectRatio:false, scales:{r:{min:0,max:100,grid:{color:'rgba(255,255,255,0.05)'}}} }
    });
  }
}

// ============================================
// ENHANCED STATE INITIALIZATION
// ============================================
state.journalEntries = JSON.parse(localStorage.getItem('sr-journal') || '[]');
state.savedTrips = JSON.parse(localStorage.getItem('sr-saved-trips') || '[]');
state.multiCityData = null;

// Save trips for comparison
function _addTripToSaved(itin) {
  if (!itin?.destination) return;
  const exists = state.savedTrips.find(t => t.destination === itin.destination && t.days === itin.days);
  if (!exists) {
    itin._savedAt = new Date().toISOString();
    state.savedTrips.push(itin);
    if (state.savedTrips.length > 10) state.savedTrips.shift();
    try { localStorage.setItem('sr-saved-trips', JSON.stringify(state.savedTrips)); } catch(e) {}
  }
}

// Enhanced DOMContentLoaded
document.addEventListener('DOMContentLoaded', () => {
  renderJournalEntries();
  setTimeout(() => { if (typeof renderBayesian === 'function') renderBayesian(); }, 200);
});
