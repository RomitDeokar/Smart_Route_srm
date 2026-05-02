# SmartRoute SRMIST - Complete Agentic AI Travel Planner v4.1

## Project Overview
- **Name**: SmartRoute SRMIST
- **Goal**: Complete multi-agent autonomous travel intelligence system using 7 operational AI agents
- **Tech Stack**: Hono (Edge) + TypeScript + Leaflet.js + Chart.js + Tailwind CSS concepts
- **Platform**: Cloudflare Pages (Edge deployment)
- **APIs**: All FREE - OpenMeteo, Overpass/OSM, OpenTripMap, Wikipedia, Nominatim (no OpenAI/Claude)

## Live URLs
- **App**: (deployed on Cloudflare Pages after `npm run deploy`)
- **API Health**: `/api/health`

## Complete Agentic AI Project List

### 7 Operational AI Agents
1. **Planner Agent** - MCTS (50 iterations) + Nearest-Neighbor TSP for itinerary optimization
2. **Weather Risk Agent** - Naive Bayes classification on OpenMeteo data (sunny/cloudy/rainy)
3. **Crowd Analyzer** - Time-of-day crowd heuristic (6am-midnight prediction)
4. **Budget Optimizer** - MDP-based reward function: R = 0.4*rating + 0.3*budget + 0.2*weather - 0.1*crowd
5. **Preference Agent** - Bayesian Beta distributions (per-category) + Dirichlet time allocation
6. **Booking Assistant** - Multi-platform search (flights, trains, hotels, cabs) with real booking URLs
7. **Explainability Agent** - MDP decision trace + POMDP belief state visualization

### AI/ML Algorithms
- **Q-Learning**: epsilon-greedy with decay (0.3 -> 0.05), Q-table persisted
- **Monte Carlo Tree Search (MCTS)**: 50 iterations with UCB1 selection for route optimization
- **Bayesian Inference**: Beta distributions for preferences with 95% CI
- **Dirichlet Distribution**: Time allocation proportions across categories
- **Naive Bayes**: Gaussian likelihood weather classification
- **POMDP**: Belief state updates over trip quality (excellent/good/average/poor)
- **MDP Reward Function**: Multi-objective optimization across satisfaction, budget, weather, crowd

### Integrated Agentic Modules
- **Interactive Leaflet Map** with day-colored routes, origin-destination lines, satellite/dark/light/street layers
- **Agentic Booking Wizard** - 8-step workflow (Plan -> Flights -> Trains -> Hotels -> Cabs -> Review -> Pay -> Confirmed)
- **Multi-City Trip Planner** - Plan across multiple cities with per-city itineraries (from TripSage)
- **Trip Comparison** - Side-by-side destination comparison table
- **Destination Recommendations** - AI-powered "Help Me Choose" with interest matching
- **Half-Day Planner** - Quick plan for remaining time
- **Emergency Replan** - Delay/Weather/Crowd-based itinerary adjustment with MDP decisions
- **Smart Packing List** - AI-curated based on duration, weather, persona (from NOMAD)
- **Travel Atlas** - World map tracking all planned trips (from NOMAD)
- **Dashboard** - Travel stats, reward progression chart, preference radar chart
- **Trip Journal** - Personal notes with date/destination tagging (from NOMAD)
- **Emergency Contacts** - City-specific emergency numbers (police, ambulance, hospital, etc.)
- **Safety Tips** - Persona + city-specific safety recommendations
- **Currency Converter** - INR/USD/EUR/GBP/JPY/THB with fallback rates
- **World Clock** - Local + destination time display
- **Language Tips** - Regional phrases for 8+ Indian languages
- **Social Discovery** - Trending spots, hidden gems (crowd < 40%), foodie spots
- **AI Chatbot** - Context-aware assistant with 10+ topic categories
- **Booking History** - Sidebar with all past bookings saved to localStorage
- **PDF Export** - Print-friendly itinerary with all details
- **Share Trip** - Web Share API or clipboard copy
- **Voice Input** - Speech recognition for destination entry
- **GPS Detection** - Auto-detect origin city via Nominatim reverse geocoding
- **Dark/Light Theme** - System-aware with manual toggle
- **Agent Communication Graph** - Canvas visualization of agent interactions
- **MDP/POMDP Flow Diagrams** - Visual state space and pipeline
- **Activity Rating** - Star ratings that update Bayesian/POMDP/Q-Learning in real-time

### Persona Modes
- Solo Traveler, Family, Luxury, Adventure (each affects budget, packing, recommendations)

### Data Sources (All FREE)
- **OpenMeteo** - Weather forecasts (16-day, hourly humidity)
- **Overpass API** - OpenStreetMap attractions, tourism data
- **OpenTripMap** - Cultural, historic, natural places
- **Wikipedia API** - Place photos and thumbnails
- **Nominatim** - Geocoding and reverse geocoding
- **exchangerate.host** - Currency conversion (with offline fallbacks)

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | System status + feature list + complete project manifest |
| GET | `/api/project-list` | Complete agentic AI project list, modules, and quality-fix manifest |
| POST | `/api/generate-trip` | Generate full itinerary |
| POST | `/api/generate-multi-city` | Multi-city trip planner |
| POST | `/api/rate` | Rate activity (updates Bayesian/POMDP/QL) |
| POST | `/api/search-flights` | Search flight options |
| POST | `/api/search-trains` | Search train options |
| POST | `/api/search-hotels` | Search hotel options |
| POST | `/api/search-cabs` | Search local transport |
| POST | `/api/recommendations` | AI destination recommendations |
| POST | `/api/compare-trips` | Compare multiple trips |
| POST | `/api/replan` | Emergency replan (delay/weather/crowd) |
| GET | `/api/nearby` | Nearby places via Overpass |
| GET | `/api/emergency-contacts` | City emergency numbers |
| GET | `/api/safety-tips` | City + persona safety tips |
| GET | `/api/ai-state` | Current AI agent state |
| POST | `/api/chat` | AI chatbot with context |

## Project Structure
```
webapp/
├── src/
│   └── index.tsx          # Hono backend (all 7 agents + APIs)
├── public/static/
│   ├── index.html         # Complete SPA UI
│   ├── app.js            # Frontend logic (1700+ lines)
│   ├── styles.css        # Full CSS (380+ lines)
│   └── style.css         # Legacy styles
├── ecosystem.config.cjs   # PM2 configuration
├── package.json
├── vite.config.ts
├── wrangler.jsonc
└── README.md
```

## Quick Start
```bash
npm install
npm run build
pm2 start ecosystem.config.cjs
# or: npx wrangler pages dev dist --ip 0.0.0.0 --port 3000
```

## Deployment
```bash
npm run build
npx wrangler pages deploy dist --project-name smartroute-srmist
```

## Reference Projects Integrated
- **Original SmartRoute** - Core 7-agent system, MDP/RL, Bayesian, POMDP
- **NOMAD** - Atlas, packing lists, drag-drop planning, dashboard, notes, file management
- **TripSage AI** - Multi-city routing, edge-first architecture, AI gateway pattern
- **Flight Finder CrewAI** - Multi-agent booking coordination concept
- **Travel Itinerary Generator** - Weather-aware itinerary, Gemini integration concept
- **Virtugo** - Map-based travel with FourSquare-style place search
- **Previous SmartRoute versions** - All UI/UX patterns, booking wizard, social discovery

## Completion / Quality Fixes in v4.1
- Added `/api/project-list` and health manifest for the complete agentic AI project list.
- Normalized `duration` and `days` request aliases so API clients cannot accidentally generate the wrong trip length.
- Made itinerary dates start-date aware instead of always mirroring weather API dates.
- Added resilient POI fallback generation so planning still works when external free APIs rate-limit or return no attractions.
- Added frontend project-list panel showing all operational agents and integrated modules.
- Hardened localStorage parsing so corrupt browser state cannot break startup.
- Escaped chatbot responses before formatting Markdown-style text to avoid unsafe HTML injection.

## Last Updated
2026-05-02
