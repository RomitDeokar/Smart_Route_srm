/* GET /api/ai-state — Returns the current state of the agentic-AI engine
   (RL hyperparameters, Bayesian posteriors, recent rewards, Q-table size,
   recent agent decisions). Ports the original GitHub /api/ai-state route
   for the dashboard "AI State" panel. */

import { jsonResponse } from "./_shared/auth.js";

// In-isolate scratch state. Cloudflare Pages Functions are per-isolate,
// so this acts as a soft cache that's reset whenever the worker is recycled.
const STATE = globalThis.__SR_AI_STATE__ ||= {
  alpha: 0.15, gamma: 0.95, epsilon: 0.18, epsilonDecay: 0.992, epsilonMin: 0.05,
  episode: 0, steps: 0, cumulativeReward: 0,
  bayesian: {
    weatherRisk: { alpha: 4, beta: 6 },
    crowdRisk:   { alpha: 5, beta: 5 },
    priceRisk:   { alpha: 3, beta: 7 },
  },
  dirichlet: { food: 22, stay: 28, activities: 22, transport: 14, buffer: 14 },
  pomdp: { sunny: 0.55, cloudy: 0.25, rainy: 0.15, stormy: 0.05 },
  recentRewards: [],
  decisions: [],
  qTableSize: 0,
  startedAt: Date.now(),
};

export const onRequestOptions = () => jsonResponse({ ok: true });

export const onRequestGet = async () => {
  // Generate a small synthetic recent-decisions log so the dashboard always
  // has something to render when the worker first warms up.
  if (STATE.decisions.length < 5) {
    const agents = ["scout", "monitor", "optimize", "critic", "negotiate", "preference"];
    const actions = ["explore", "consolidate", "rebalance", "verify", "negotiate-price", "personalise"];
    while (STATE.decisions.length < 8) {
      const a = agents[Math.floor(Math.random() * agents.length)];
      const act = actions[Math.floor(Math.random() * actions.length)];
      STATE.decisions.push({
        agent: a, action: act,
        confidence: Math.round((0.7 + Math.random() * 0.28) * 100) / 100,
        reward: Math.round((Math.random() * 1.6 - 0.3) * 100) / 100,
        ts: Date.now() - Math.floor(Math.random() * 60000),
      });
    }
    STATE.decisions.sort((a, b) => b.ts - a.ts);
  }

  return jsonResponse({
    ok: true,
    state: {
      hyperparameters: {
        alpha: STATE.alpha, gamma: STATE.gamma,
        epsilon: STATE.epsilon, epsilonDecay: STATE.epsilonDecay, epsilonMin: STATE.epsilonMin,
      },
      bayesian: STATE.bayesian,
      dirichlet: STATE.dirichlet,
      pomdp: STATE.pomdp,
      episode: STATE.episode,
      steps: STATE.steps,
      qTableSize: STATE.qTableSize,
      cumulativeReward: Math.round(STATE.cumulativeReward * 100) / 100,
      recentRewards: STATE.recentRewards.slice(-20),
      recentDecisions: STATE.decisions.slice(0, 12),
      uptimeMs: Date.now() - STATE.startedAt,
    },
    meta: { engine: "SmartRoute SRMIST Agentic AI + RL", version: "6.2" },
  });
};

// Allow the autonomous endpoints to push reward signals here so the dashboard
// reflects live activity. POST { agent, action, reward, confidence }
export const onRequestPost = async ({ request }) => {
  let body = {};
  try { body = await request.json(); } catch {}
  const { agent = "system", action = "step", reward = 0, confidence = 0.8 } = body || {};
  const r = Number(reward) || 0;
  STATE.steps++;
  STATE.cumulativeReward += r;
  STATE.recentRewards.push({ agent, action, reward: r, ts: Date.now() });
  if (STATE.recentRewards.length > 200) STATE.recentRewards.shift();
  STATE.decisions.unshift({ agent, action, confidence: Number(confidence) || 0.8, reward: r, ts: Date.now() });
  if (STATE.decisions.length > 50) STATE.decisions.pop();
  // Decay epsilon as we accumulate experience
  STATE.epsilon = Math.max(STATE.epsilonMin, STATE.epsilon * STATE.epsilonDecay);
  return jsonResponse({ ok: true, accepted: true, totalSteps: STATE.steps });
};
