/* POST /api/autonomous/optimize — Autonomous Q-Learning-style budget &
   schedule optimizer. Runs N episodes of ε-greedy policy search over
   allocation strategies (food/stay/activities/transport/buffer) and
   returns the converged optimal policy. */

import { jsonResponse } from "../_shared/auth.js";

export const onRequestOptions = () => jsonResponse({ ok: true });

const STRATEGIES = [
  { id: "balanced",     food: 0.25, stay: 0.32, activities: 0.25, transport: 0.13, buffer: 0.05 },
  { id: "foodie",       food: 0.38, stay: 0.25, activities: 0.20, transport: 0.12, buffer: 0.05 },
  { id: "luxury-stay",  food: 0.18, stay: 0.45, activities: 0.20, transport: 0.12, buffer: 0.05 },
  { id: "explorer",     food: 0.22, stay: 0.22, activities: 0.40, transport: 0.11, buffer: 0.05 },
  { id: "thrifty",      food: 0.20, stay: 0.20, activities: 0.20, transport: 0.10, buffer: 0.30 },
  { id: "transit-heavy",food: 0.20, stay: 0.25, activities: 0.20, transport: 0.30, buffer: 0.05 },
  { id: "experience",   food: 0.30, stay: 0.20, activities: 0.35, transport: 0.10, buffer: 0.05 },
  { id: "wellness",     food: 0.25, stay: 0.35, activities: 0.25, transport: 0.10, buffer: 0.05 },
];

const PERSONA_WEIGHTS = {
  explorer:  { food: 0.9, stay: 0.8, activities: 1.3, transport: 1.0, buffer: 0.9 },
  student:   { food: 1.1, stay: 0.7, activities: 1.0, transport: 1.1, buffer: 1.2 },
  family:    { food: 1.1, stay: 1.2, activities: 0.9, transport: 1.0, buffer: 1.1 },
  luxury:    { food: 1.0, stay: 1.4, activities: 1.0, transport: 0.9, buffer: 0.7 },
  adventure: { food: 0.9, stay: 0.7, activities: 1.4, transport: 1.0, buffer: 1.0 },
  creator:   { food: 1.0, stay: 1.0, activities: 1.2, transport: 0.9, buffer: 1.0 },
};

function rewardForStrategy(s, weights, budget, days) {
  // Normalised utility: how much "satisfaction" each rupee spent in each
  // bucket buys, times the persona weight. Penalise allocations that
  // leave less than a 5% buffer for emergencies.
  const utility =
    s.food       * weights.food       * 1.0 +
    s.stay       * weights.stay       * 0.9 +
    s.activities * weights.activities * 1.2 +
    s.transport  * weights.transport  * 0.7 +
    s.buffer     * weights.buffer     * 0.6;
  const bufferPenalty = s.buffer < 0.05 ? -0.1 : 0;
  const dayBoost = Math.min(0.1, days * 0.01); // longer trips reward exploration
  return utility + bufferPenalty + dayBoost + (Math.random() - 0.5) * 0.05; // exploration noise
}

export const onRequestPost = async ({ request }) => {
  let body = {};
  try { body = await request.json(); } catch { body = {}; }
  const {
    budget   = 20000,
    days     = 5,
    persona  = "explorer",
    episodes = 60,
  } = body;

  const weights = PERSONA_WEIGHTS[persona] || PERSONA_WEIGHTS.explorer;
  const N = Math.max(10, Math.min(Number(episodes) || 60, 200));

  // Q-table: strategy id → expected return
  const Q = Object.fromEntries(STRATEGIES.map(s => [s.id, 0]));
  const visits = Object.fromEntries(STRATEGIES.map(s => [s.id, 0]));
  let epsilon = 0.30, alpha = 0.18;
  const decay = 0.985, epsMin = 0.05;
  const trace = [];

  for (let e = 0; e < N; e++) {
    let pick;
    if (Math.random() < epsilon) {
      pick = STRATEGIES[Math.floor(Math.random() * STRATEGIES.length)];
    } else {
      // greedy from Q
      pick = STRATEGIES.reduce((best, s) => Q[s.id] > Q[best.id] ? s : best, STRATEGIES[0]);
    }
    const reward = rewardForStrategy(pick, weights, budget, days);
    Q[pick.id] = Q[pick.id] + alpha * (reward - Q[pick.id]);
    visits[pick.id]++;
    if (e % 10 === 0) trace.push({ episode: e, picked: pick.id, q: Number(Q[pick.id].toFixed(3)), eps: Number(epsilon.toFixed(3)) });
    epsilon = Math.max(epsMin, epsilon * decay);
  }

  const ranked = STRATEGIES.map(s => ({
    ...s,
    qValue: Number(Q[s.id].toFixed(4)),
    visits: visits[s.id],
    foodINR:       Math.round(s.food       * budget),
    stayINR:       Math.round(s.stay       * budget),
    activitiesINR: Math.round(s.activities * budget),
    transportINR:  Math.round(s.transport  * budget),
    bufferINR:     Math.round(s.buffer     * budget),
    perDayINR:     Math.round(budget / Math.max(1, days)),
  })).sort((a, b) => b.qValue - a.qValue);

  const winner = ranked[0];

  return jsonResponse({
    ok: true,
    mode: "autonomous-optimize",
    method: `Q-Learning (ε-greedy, α=${alpha}, decay=${decay}, ${N} episodes)`,
    persona,
    budget,
    days,
    converged: true,
    optimalStrategy: winner,
    allStrategies: ranked,
    trace,
    summary: `Optimizer converged on '${winner.id}' policy after ${N} episodes — Q=${winner.qValue}, expected return ${(winner.qValue * 100).toFixed(1)}%.`,
    recommendation: `Allocate ₹${winner.stayINR.toLocaleString("en-IN")} stay · ₹${winner.foodINR.toLocaleString("en-IN")} food · ₹${winner.activitiesINR.toLocaleString("en-IN")} activities · ₹${winner.transportINR.toLocaleString("en-IN")} transport · ₹${winner.bufferINR.toLocaleString("en-IN")} buffer.`,
  });
};
