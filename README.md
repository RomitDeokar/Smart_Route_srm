# SmartRoute SRMIST v6.0 — Autonomous Agentic AI Travel Platform

> Multi-agent autonomous AI travel planner. React + Cloudflare Pages Functions.
> The working GitHub backend (7-agent RL engine) merged with a brand-new React UI,
> JWT auth, user dashboards and an autonomous-pilot pipeline.

---

## What's new in v6.0

* **Brand-new React UI** (Plus Jakarta Sans + Sora, Framer Motion, light/blue SaaS look)
* **Login / Register pages** with JWT auth (HS256, Web Crypto only)
* **User dashboard** with hero, live map, AI sync feed, curated destinations
* **Sidebar AppShell** with 8 pages (Dashboard, Map, Budget, Itinerary, Atlas, Packing, AI, Reservations)
* **Autonomous agent fleet (15 agents)** including new:
  * `optimize` — Q-Learning ε-greedy budget allocator (8 strategies, 60–200 episodes)
  * `autopilot` — End-to-end orchestrator that runs Scout → Monitor → Optimize → Itinerary → Critic → Negotiate
* **All 7 original RL/AI agents preserved** (MCTS, Q-Learning, Bayesian Thompson, POMDP, Naive Bayes)
* **Full city database** (45+ Indian cities + SRM campuses) with curated POIs and coordinates

---

## Live URLs

| Service | URL |
|---------|-----|
| Local dev    | <http://localhost:3000> |
| Health check | `/api/health` |

---

## Quick Start

```bash
npm install
npm run build
npx wrangler pages dev dist --ip 0.0.0.0 --port 3000
# or use PM2: pm2 start ecosystem.config.cjs
```

Demo login: any email + any password → instant JWT, or click **Continue as Demo User**.

---

## Architecture

```
webapp/
├── src/                          ← React + Vite frontend
│   ├── App.jsx                   ← React Router + auth + toasts + splash
│   ├── main.jsx
│   ├── styles.css                ← 2000+ line design system
│   ├── layout/AppShell.jsx       ← Sidebar + topbar + chat fab
│   ├── pages/
│   │   ├── Dashboard.jsx         ← Main user dashboard (1000+ lines)
│   │   ├── MapExplorer.jsx
│   │   ├── Budget.jsx
│   │   ├── Itinerary.jsx         ← Persona, day timeline, AI plans
│   │   ├── AIAssistant.jsx
│   │   ├── Reservations.jsx
│   │   ├── Atlas.jsx             ← Travel atlas / world tracker
│   │   ├── Packing.jsx           ← Smart packing list
│   │   ├── Login.jsx             ← Auth — branded + JWT
│   │   └── Register.jsx
│   ├── components/               ← 35+ components (Map, Chat, Charts, Pipeline, …)
│   ├── hooks/                    ← useAuth, useAgentStream
│   └── lib/api.js                ← Robust fetch wrapper (handles empty/non-JSON)
│
├── functions/api/                ← Cloudflare Pages Functions backend
│   ├── _shared/                  ← Auth helpers, city data, planner, hotels, transport
│   │   ├── auth.js               ← HS256 JWT (Web Crypto)
│   │   ├── cities.js             ← 45+ cities, 200+ POIs with coordinates
│   │   ├── planner.js            ← Multi-agent plan builder
│   │   ├── hotels-real.js        ← Realistic hotel inventory + booking URLs
│   │   ├── transport.js          ← Flights/trains generator
│   │   ├── extras.js             ← Restaurants, language tips, packing, safety
│   │   └── srm.js                ← SRM-specific campus data
│   ├── auth/{login,register,me}.js
│   ├── autonomous/
│   │   ├── status.js             ← 13-agent fleet heartbeat
│   │   ├── scout.js              ← Persona-weighted attraction scoring
│   │   ├── monitor.js            ← Weather + booking + crowd watchpoints
│   │   ├── critic.js             ← Rubric-weighted self-audit
│   │   ├── negotiate.js          ← Bayesian discount + 5-round bargaining
│   │   ├── replan.js             ← Recovery / contingency planner
│   │   ├── optimize.js           ← NEW · Q-Learning budget optimizer
│   │   └── autopilot.js          ← NEW · End-to-end orchestrator
│   ├── itinerary.js              ← Day-by-day plan + real attractions + weather
│   ├── plan.js                   ← Multi-agent mock-AI plan
│   ├── chat.js                   ← Context-aware chatbot
│   ├── nearby.js, recommendations.js, packing-list.js, …
│   ├── flights/search.js, hotels/search.js, trains/search.js, cabs/search.js
│   ├── budget/{create,update,suggest,status}.js
│   └── payments/checkout.js
│
├── index.html                    ← Vite entry
├── vite.config.ts                ← React plugin + dev proxy
├── wrangler.jsonc                ← Cloudflare Pages config
└── ecosystem.config.cjs          ← PM2 config (wrangler pages dev)
```

---

## API Endpoints

### Auth
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/auth/register` | Create account, returns JWT |
| POST | `/api/auth/login`    | Login, returns JWT (auto-provisions across edge isolates) |
| GET  | `/api/auth/me`       | Current user from Bearer token |

### Trip planning
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/itinerary`           | Full structured day-by-day plan with weather + transport |
| POST | `/api/plan`                | Multi-agent mock-AI plan |
| POST | `/api/chat`                | Context-aware AI chatbot |
| GET  | `/api/nearby`              | Overpass/OpenTripMap nearby places |
| POST | `/api/recommendations`     | AI destination recommendations |
| POST | `/api/restaurants`         | Restaurant suggestions |
| POST | `/api/quick-trip`          | GPS-based half-day plan |
| POST | `/api/risk-score`          | AI-powered travel risk |
| POST | `/api/packing-list`        | AI packing list |
| POST | `/api/crowd-info`          | Crowd density predictions |
| POST | `/api/emergency-options`   | Emergency replan |
| GET  | `/api/safety-tips`         | City + persona safety tips |
| GET  | `/api/language-tips`       | Regional Indian phrases |

### Bookings & payments
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/flights/search`   | Realistic flight search |
| POST | `/api/trains/search`    | Train search |
| POST | `/api/hotels/search`    | Hotel search with booking URLs |
| POST | `/api/cabs/search`      | Local transport |
| POST | `/api/activities/search`| Activities |
| POST | `/api/budget/create`    | Create budget |
| POST | `/api/budget/update`    | Log expense |
| GET  | `/api/budget/suggest`   | AI budget suggestions |
| GET  | `/api/budget/status`    | Budget status |
| POST | `/api/payments/checkout`| Stripe checkout session |

### Autonomous agents
| Method | Path | Description |
|--------|------|-------------|
| GET/POST | `/api/autonomous/status`     | 15-agent fleet heartbeat |
| POST     | `/api/autonomous/scout`      | Hidden gems + persona ranking |
| POST     | `/api/autonomous/monitor`    | Live watchpoints + alerts |
| POST     | `/api/autonomous/critic`     | Rubric-weighted self-audit |
| POST     | `/api/autonomous/negotiate`  | Bayesian discount bargaining |
| POST     | `/api/autonomous/replan`     | Recovery / contingency planner |
| POST     | `/api/autonomous/optimize`   | **NEW** — Q-Learning budget allocator |
| POST     | `/api/autonomous/autopilot`  | **NEW** — End-to-end pipeline orchestrator |

---

## AI / RL Algorithms

| Algorithm | Where | Details |
|-----------|-------|---------|
| Q-Learning (ε-greedy)        | `autonomous/optimize.js` | 60–200 episodes, decay=0.985, α=0.18 |
| Thompson Sampling (Beta)     | `_shared/planner.js`     | Per-category preference posterior |
| MCTS + UCB1                  | `_shared/planner.js`     | 200 iterations route planner |
| MDP value iteration          | `_shared/planner.js`     | Policy decision for activity scheduling |
| Naive Bayes (Bernoulli)      | `_shared/planner.js`     | Weather risk classification |
| Gaussian-Process (RBF)       | `_shared/planner.js`     | Crowd density surrogate |
| Bayesian discount prior      | `autonomous/negotiate.js`| 5-round iterative bargaining |
| Rubric-weighted audit        | `autonomous/critic.js`   | 6-criterion self-critique |
| Persona-weighted scoring     | `autonomous/scout.js`    | Type/description matching |

---

## Deployment (Cloudflare Pages)

```bash
npm run build
npx wrangler pages deploy dist --project-name smartroute-srmist
```

The `dist/` folder is the static React build; `functions/` is auto-detected
by Cloudflare Pages and exposed under `/api/*` as edge functions. JWT secret
can be configured via the dashboard or `wrangler secret put JWT_SECRET`.

---

## Bug fixes vs previous releases

* Robust `safeJson` fetch wrapper — never throws on empty / non-JSON
* Login auto-provisions across edge isolates (per-isolate user store)
* Itinerary anchored to destination coords (no more 100 km off-map markers)
* Distance filter (≤ 120 km) on attractions to drop fuzzy-matched outliers
* Restaurants sorted by haversine distance from destination
* `register` is idempotent if same email+password is re-submitted
* All API routes wrap JSON parsing in try/catch (no 500s on bad bodies)
* Edge-safe (no Node.js Buffer / fs / Stripe SDK — all Web Crypto)

---

## Last Updated
2026-05-04 — v6.0 (React UI + JWT auth + autopilot + optimize agents)
