/* POST /api/rate — Accept a user rating for a generated trip and feed it
   back into the AI state's reward signal. Ports /api/rate from the
   original GitHub backend. */

import { jsonResponse } from "./_shared/auth.js";

const STATE = globalThis.__SR_AI_STATE__ ||= {
  alpha: 0.15, gamma: 0.95, epsilon: 0.18, epsilonDecay: 0.992, epsilonMin: 0.05,
  episode: 0, steps: 0, cumulativeReward: 0,
  bayesian: { weatherRisk:{alpha:4,beta:6}, crowdRisk:{alpha:5,beta:5}, priceRisk:{alpha:3,beta:7} },
  dirichlet: { food:22, stay:28, activities:22, transport:14, buffer:14 },
  pomdp: { sunny:0.55, cloudy:0.25, rainy:0.15, stormy:0.05 },
  recentRewards: [], decisions: [], qTableSize: 0, startedAt: Date.now(),
  ratings: [],
};

export const onRequestOptions = () => jsonResponse({ ok: true });

export const onRequestPost = async ({ request }) => {
  let body = {};
  try { body = await request.json(); } catch {}
  const { rating, destination, persona, comment } = body || {};
  const r = Number(rating);
  if (!Number.isFinite(r) || r < 1 || r > 5) {
    return jsonResponse({ ok: false, error: "rating must be a number 1-5" }, 400);
  }

  // Reward = (rating - 3) / 2  →  -1 to +1
  const reward = (r - 3) / 2;
  STATE.episode++;
  STATE.cumulativeReward += reward;
  STATE.recentRewards.push({ agent: "user-feedback", action: "rate", reward, ts: Date.now() });
  if (STATE.recentRewards.length > 200) STATE.recentRewards.shift();

  // Update Bayesian posteriors based on rating
  if (r >= 4) {
    STATE.bayesian.weatherRisk.alpha++;
    STATE.bayesian.crowdRisk.alpha++;
  } else if (r <= 2) {
    STATE.bayesian.weatherRisk.beta++;
    STATE.bayesian.crowdRisk.beta++;
  }

  STATE.ratings ||= [];
  STATE.ratings.push({
    rating: r, destination: destination || null, persona: persona || null,
    comment: (comment || "").slice(0, 280), ts: Date.now(),
  });
  if (STATE.ratings.length > 100) STATE.ratings.shift();

  STATE.decisions.unshift({
    agent: "user-feedback", action: `rated ${r}★`,
    confidence: r / 5, reward, ts: Date.now(),
  });
  if (STATE.decisions.length > 50) STATE.decisions.pop();

  const avg = STATE.ratings.reduce((s, x) => s + x.rating, 0) / STATE.ratings.length;

  return jsonResponse({
    ok: true,
    accepted: true,
    rewardSignal: reward,
    cumulativeReward: Math.round(STATE.cumulativeReward * 100) / 100,
    averageRating: Math.round(avg * 100) / 100,
    totalRatings: STATE.ratings.length,
    message: r >= 4 ? "Thanks! AI agents will reinforce this strategy." :
             r <= 2 ? "Got it — agents will explore alternative strategies next time." :
                      "Thanks for the feedback — calibrating preferences.",
  });
};
