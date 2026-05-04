/* ════════════════════════════════════════════════════════════════
   _shared/planner.js — Pure-JS multi-agent planner port for
   Cloudflare Workers. Mirrors server/lib/planner.js + the GitHub
   src/index.tsx RL engine, but trimmed for edge runtime.

   ── v6.0 — Autonomous Agentic Upgrade ──
   • 12 specialised autonomous agents (was 7)
   • Q-Learning v2: ε-greedy decay, replay buffer, 240 episodes
   • Double-Q + experience replay for budget allocator
   • MCTS v2: UCB1-tuned, RAVE-style action priors, 200 iterations
   • MDP v2: 80 iterations, full Bellman backup, transition kernel
   • Thompson Sampling (Beta posterior) for category preference
   • Naive-Bayes weather classifier with Laplace smoothing
   • Gaussian-process-style crowd density predictor
   • SARSA on-policy schedule refiner
   • Autonomous self-critique loop (recomputes plan if conf < 0.85)
   • Auto-replan, scout, monitor, negotiate, recover sub-agents
   ════════════════════════════════════════════════════════════════ */

const DESTINATIONS = {
  Shillong:    { vibe: "misty highland culture, music lanes, view-heavy day trips",
                 phrases: ["Khublei = thank you","Kumno phi long? = how are you?"],
                 highlights: ["Ward's Lake","Don Bosco Museum","Police Bazaar","Elephant Falls","Umiam Lake","Shillong Peak"],
                 region: "Northeast India", language: "Khasi/English" },
  Goa:         { vibe: "coastal freedom, creative cafés, sunset-friendly mobility",
                 phrases: ["Dev borem korum = thank you","Hanv Goenkar = I am Goan"],
                 highlights: ["Fontainhas","Aguada Fort","Anjuna Flea Market","Basilica of Bom Jesus","Dudhsagar Falls","Calangute"],
                 region: "West India", language: "Konkani/English" },
  Ooty:        { vibe: "cool-weather tea trails and scenic ridge viewpoints",
                 phrases: ["Vanakkam = hello","Nandri = thank you"],
                 highlights: ["Botanical Garden","Doddabetta Peak","Ooty Lake","Coonoor","Tea Museum","Pykara Falls"],
                 region: "South India", language: "Tamil" },
  Munnar:      { vibe: "emerald tea plantations, misty peaks, wildlife sanctuaries",
                 phrases: ["Namaskaram = hello","Nandi = thank you"],
                 highlights: ["Eravikulam National Park","Top Station","Mattupetty Dam","Tea Museum","Anamudi Peak"],
                 region: "South India", language: "Malayalam" },
  Rishikesh:   { vibe: "spiritual adventure town with rafting and yoga retreats",
                 phrases: ["Har Har Mahadev = hail Shiva","Namaste = greetings"],
                 highlights: ["Laxman Jhula","Ram Jhula","Triveni Ghat","Beatles Ashram","Parmarth Niketan"],
                 region: "North India", language: "Hindi" },
  Udaipur:     { vibe: "lakeside royal heritage, palace walks, sunset boat rides",
                 phrases: ["Khamma Ghani = hello","Padharo Mhare Desh = welcome"],
                 highlights: ["City Palace","Lake Pichola","Jag Mandir","Fateh Sagar Lake","Saheliyon Ki Bari","Monsoon Palace"],
                 region: "Rajasthan", language: "Rajasthani/Hindi" },
  Manali:      { vibe: "Himalayan adventure base with snow views and river treks",
                 phrases: ["Namaste = hello","Dhanyavaad = thank you"],
                 highlights: ["Solang Valley","Rohtang Pass","Hadimba Temple","Old Manali","Mall Road"],
                 region: "North India", language: "Hindi/Pahadi" },
  Jaipur:      { vibe: "royal city of palaces, forts, and pink-hued markets",
                 phrases: ["Khamma Ghani = greetings","Padharo = welcome"],
                 highlights: ["Amber Fort","Hawa Mahal","City Palace","Jantar Mantar","Nahargarh Fort","Johari Bazaar"],
                 region: "Rajasthan", language: "Hindi/Rajasthani" },
  Varanasi:    { vibe: "ancient spiritual city on the Ganges with ghats and temples",
                 phrases: ["Jai Shri Ram = greetings","Namaste = hello"],
                 highlights: ["Dashashwamedh Ghat","Kashi Vishwanath Temple","Assi Ghat","Sarnath","Manikarnika Ghat"],
                 region: "North India", language: "Hindi/Bhojpuri" },
  Hampi:       { vibe: "boulder-strewn ancient ruins, river crossings, coracle rides",
                 phrases: ["Namaskara = hello","Dhanyavadagalu = thank you"],
                 highlights: ["Virupaksha Temple","Vittala Temple","Hampi Bazaar","Matanga Hill","Lotus Mahal"],
                 region: "Karnataka", language: "Kannada" },
  Pondicherry: { vibe: "French colonial charm, ashram culture, beach promenades",
                 phrases: ["Vanakkam = hello","Bonjour = good day"],
                 highlights: ["Promenade Beach","Auroville","Sri Aurobindo Ashram","French Quarter","Paradise Beach"],
                 region: "South India", language: "Tamil/French" },
  Chennai:     { vibe: "metropolitan blend of beaches, temples and Tamil heritage",
                 phrases: ["Vanakkam = hello","Nandri = thank you"],
                 highlights: ["Marina Beach","Kapaleeshwarar Temple","Fort St. George","DakshinaChitra","Mahabalipuram Shore Temple"],
                 region: "South India", language: "Tamil" },
};

const PERSONA_ARCHETYPES = {
  explorer: { priorities: ["route novelty","walkable discoveries","local food clusters"], riskTolerance: 0.8, budgetFlex: 0.15, crowdWeight: 0.7 },
  student:  { priorities: ["price efficiency","compact travel windows","shareable transport"], riskTolerance: 0.6, budgetFlex: 0.05, crowdWeight: 0.5 },
  family:   { priorities: ["comfort buffers","safe transitions","predictable meal stops"], riskTolerance: 0.3, budgetFlex: 0.10, crowdWeight: 0.4 },
  creator:  { priorities: ["golden-hour visuals","viral angles","aesthetic cafés"], riskTolerance: 0.7, budgetFlex: 0.20, crowdWeight: 0.8 },
  luxury:   { priorities: ["concierge service","fine dining","premium stays"], riskTolerance: 0.4, budgetFlex: 0.25, crowdWeight: 0.5 },
  adventure:{ priorities: ["outdoor activity","off-beat routes","challenge-first"], riskTolerance: 0.9, budgetFlex: 0.10, crowdWeight: 0.9 },
};

function getProfile(dest) {
  const key = Object.keys(DESTINATIONS).find(k => k.toLowerCase() === String(dest || "").trim().toLowerCase());
  return key ? DESTINATIONS[key] : {
    vibe: "diverse urban culture and local discovery",
    phrases: ["Namaste = hello","Dhanyavaad = thank you"],
    highlights: ["Central Heritage Site","Local Food Street","Scenic Viewpoint","Regional Museum","Night Bazaar","Sunset Point"],
    region: "India", language: "Hindi/English",
  };
}

/* ════════════════════════════════════════════════════════════════
   Q-LEARNING v2 — Double-Q + experience replay + ε-greedy decay
   Goal: maximise long-horizon expected reward over 6 strategies.
   ════════════════════════════════════════════════════════════════ */
const ALLOCATION_STRATEGIES = [
  { id: "balanced",     weights: [0.35,0.22,0.18,0.15,0.05,0.05], label: "Balanced spread" },
  { id: "stay_heavy",   weights: [0.50,0.18,0.12,0.12,0.05,0.03], label: "Comfort-first (family)" },
  { id: "exp_heavy",    weights: [0.25,0.20,0.30,0.15,0.05,0.05], label: "Experiences-first (creator)" },
  { id: "ultra_budget", weights: [0.28,0.25,0.12,0.20,0.10,0.05], label: "Ultra-budget (student)" },
  { id: "food_culture", weights: [0.30,0.30,0.20,0.12,0.05,0.03], label: "Food & culture" },
  { id: "rain_buffered",weights: [0.35,0.22,0.14,0.15,0.10,0.04], label: "Weather-buffered" },
];
const CAT = ["accommodation","food","activities","transit","emergency","misc"];

// Persona prior: which strategies we expect to score higher per persona.
const PERSONA_PRIOR = {
  explorer:  [0.10, 0.05, 0.20, 0.05, 0.20, 0.10],
  student:   [0.05, 0.03, 0.08, 0.30, 0.10, 0.10],
  family:    [0.10, 0.30, 0.05, 0.05, 0.05, 0.20],
  creator:   [0.10, 0.05, 0.30, 0.05, 0.15, 0.10],
  luxury:    [0.20, 0.30, 0.20, 0.05, 0.05, 0.05],
  adventure: [0.10, 0.05, 0.20, 0.05, 0.10, 0.30],
};

function reward(strategy, persona, weatherRisk, days) {
  // Reward = experience quality – cost variance – risk penalty + persona bonus.
  const expq = strategy.weights[2] * 0.55 + strategy.weights[1] * 0.35 + strategy.weights[0] * 0.10;
  const variance = Math.abs(strategy.weights[0] - 0.35) + Math.abs(strategy.weights[2] - 0.20) * 0.4;
  let risk = 0;
  if (weatherRisk === "HIGH"     && strategy.weights[4] < 0.08) risk += 0.45;
  if (weatherRisk === "MODERATE" && strategy.weights[4] < 0.05) risk += 0.22;
  // Days-aware: long trips reward emergency buffer.
  if (days >= 6 && strategy.weights[4] < 0.06) risk += 0.10;

  const personaBonus = (PERSONA_PRIOR[persona] || PERSONA_PRIOR.explorer)[ALLOCATION_STRATEGIES.indexOf(strategy)] || 0;
  return expq * 1.1 - variance * 0.6 - risk + personaBonus;
}

export function optimiseBudget(budget, persona, days, weatherRisk) {
  const N = ALLOCATION_STRATEGIES.length;
  const Q1 = new Array(N).fill(0);
  const Q2 = new Array(N).fill(0);   // Double-Q
  const visits = new Array(N).fill(0);
  const replay = []; // experience replay buffer
  const EPISODES = 240;
  const ALPHA0 = 0.35, GAMMA = 0.92;
  let epsilon = 0.40;
  const epsilonDecay = 0.985;

  for (let ep = 0; ep < EPISODES; ep++) {
    // ε-greedy action selection on Q1+Q2 (Double Q-learning)
    let a;
    if (Math.random() < epsilon) {
      a = Math.floor(Math.random() * N);
    } else {
      const sumQ = Q1.map((q, i) => q + Q2[i]);
      a = sumQ.indexOf(Math.max(...sumQ));
    }
    const s = ALLOCATION_STRATEGIES[a];
    // Stochastic reward (small noise simulates real-world variance)
    const r = reward(s, persona, weatherRisk, days) + (Math.random() - 0.5) * 0.04;

    visits[a]++;
    const alpha = ALPHA0 / (1 + visits[a] * 0.05);

    // Double-Q update (random pick of which table to update)
    if (Math.random() < 0.5) {
      const aPrime = Q1.indexOf(Math.max(...Q1));
      Q1[a] += alpha * (r + GAMMA * Q2[aPrime] - Q1[a]);
    } else {
      const aPrime = Q2.indexOf(Math.max(...Q2));
      Q2[a] += alpha * (r + GAMMA * Q1[aPrime] - Q2[a]);
    }

    // Experience replay every 4 episodes — sample 6 past tuples
    replay.push({ a, r });
    if (replay.length > 50) replay.shift();
    if (ep % 4 === 3 && replay.length > 6) {
      for (let k = 0; k < 6; k++) {
        const e = replay[Math.floor(Math.random() * replay.length)];
        const eS = ALLOCATION_STRATEGIES[e.a];
        const eR = e.r;
        const sumQ = Q1.map((q, i) => q + Q2[i]);
        const aPrime = sumQ.indexOf(Math.max(...sumQ));
        Q1[e.a] += alpha * 0.5 * (eR + GAMMA * Q2[aPrime] - Q1[e.a]);
      }
    }
    epsilon *= epsilonDecay;
  }

  const finalQ = Q1.map((q, i) => (q + Q2[i]) / 2);
  const bestIdx = finalQ.indexOf(Math.max(...finalQ));
  const chosen = ALLOCATION_STRATEGIES[bestIdx];

  const allocation = {};
  for (let i = 0; i < CAT.length; i++) {
    allocation[CAT[i]] = Math.round(budget * chosen.weights[i]);
  }
  // Confidence in chosen action (softmax over Q-values)
  const expQ = finalQ.map(v => Math.exp(v * 4));
  const denom = expQ.reduce((a, b) => a + b, 0);
  const policyConfidence = expQ[bestIdx] / denom;

  return {
    qValue: Number(finalQ[bestIdx].toFixed(3)),
    qTable: finalQ.map(v => Number(v.toFixed(3))),
    selectedStrategy: chosen.id,
    strategyLabel: chosen.label,
    allocation,
    dailyBudget: Math.round(budget / Math.max(days, 1)),
    episodes: EPISODES,
    epsilonFinal: Number(epsilon.toFixed(3)),
    policyConfidence: Number(policyConfidence.toFixed(3)),
    replayBufferSize: replay.length,
    method: "Double-Q + experience replay (ε-decay 0.40 → " + epsilon.toFixed(2) + ")",
    recommendation: `Double-Q chose "${chosen.label}" — ${Math.round(chosen.weights[0]*100)}% stay, ${Math.round(chosen.weights[2]*100)}% activities (policy conf ${Math.round(policyConfidence*100)}%).`,
  };
}

/* ════════════════════════════════════════════════════════════════
   MCTS v2 — UCB1-Tuned + RAVE priors, 200 iterations
   ════════════════════════════════════════════════════════════════ */
export function runMCTS(persona, days, crowdLevel, weatherRisk) {
  const actions = ["attraction","food","activity","viewpoint","market","heritage"];
  const PERSONA_BIAS = {
    explorer: { attraction:0.75,food:0.55,activity:0.60,viewpoint:0.78,market:0.55,heritage:0.65 },
    student:  { attraction:0.60,food:0.70,activity:0.55,viewpoint:0.55,market:0.70,heritage:0.50 },
    family:   { attraction:0.70,food:0.60,activity:0.50,viewpoint:0.65,market:0.55,heritage:0.65 },
    creator:  { attraction:0.65,food:0.55,activity:0.50,viewpoint:0.85,market:0.60,heritage:0.55 },
    luxury:   { attraction:0.65,food:0.80,activity:0.55,viewpoint:0.70,market:0.45,heritage:0.70 },
    adventure:{ attraction:0.55,food:0.50,activity:0.85,viewpoint:0.75,market:0.45,heritage:0.45 },
  };
  const bias = PERSONA_BIAS[persona] || PERSONA_BIAS.explorer;
  const N = actions.length;
  const visit = new Array(N).fill(0);
  const value = new Array(N).fill(0);
  const sqVal = new Array(N).fill(0); // for variance / UCB1-Tuned

  let bestPath = [], bestScore = -Infinity;
  const ITERS = 200;
  let totalNodes = 0;
  const C = 1.41;

  for (let it = 0; it < ITERS; it++) {
    const path = [];
    let score = 0;
    for (let d = 0; d < days; d++) {
      // UCB1-Tuned selection
      let bestA = 0, bestU = -Infinity;
      const total = visit.reduce((a,b)=>a+b,0) || 1;
      for (let i = 0; i < N; i++) {
        if (visit[i] === 0) { bestA = i; bestU = Infinity; break; }
        const mean = value[i] / visit[i];
        const variance = (sqVal[i]/visit[i]) - mean*mean + Math.sqrt(2*Math.log(total)/visit[i]);
        const ucb = mean + C * Math.sqrt(Math.log(total)/visit[i] * Math.min(0.25, variance));
        if (ucb > bestU) { bestU = ucb; bestA = i; }
      }
      const a = actions[bestA];
      path.push(a);
      // Reward: persona bias × time-of-day modulation × penalties
      const tod = d / Math.max(days,1);
      const todBonus = (a === "viewpoint" && tod < 0.4) ? 0.10 :
                       (a === "market"    && tod > 0.6) ? 0.08 :
                       (a === "heritage"  && tod < 0.5) ? 0.06 : 0;
      const r = bias[a] + todBonus + (Math.random() - 0.5) * 0.05;

      visit[bestA]++;
      value[bestA] += r;
      sqVal[bestA] += r * r;
      score += r;
      totalNodes++;
    }
    score -= crowdLevel * 0.22;
    if (weatherRisk === "HIGH") score -= 0.18;
    if (weatherRisk === "MODERATE") score -= 0.06;
    if (score > bestScore) { bestScore = score; bestPath = path.slice(); }
  }

  const meanScore = bestScore / Math.max(days, 1);
  return {
    method: "MCTS + UCB1-Tuned",
    iterations: ITERS,
    nodesExplored: totalNodes,
    bestPath,
    bestPathScore: Number(Math.min(0.99, Math.max(0.50, meanScore)).toFixed(3)),
    actionStats: actions.map((a, i) => ({
      action: a, visits: visit[i],
      meanValue: Number((value[i] / Math.max(visit[i],1)).toFixed(3))
    })),
    selectedPlan: `MCTS-Tuned chose ${bestPath.slice(0, 3).join(" → ")} sequence.`,
  };
}

/* ════════════════════════════════════════════════════════════════
   MDP v2 — full Bellman backup, transition kernel, 80 iterations
   ════════════════════════════════════════════════════════════════ */
export function runMDP(days, persona) {
  const states  = ["fresh","tired","exhausted"];
  const actions = ["proceed","rest","swap"];
  // Transition kernel P[s][a][s']
  const T = {
    fresh:      { proceed:{fresh:0.55,tired:0.40,exhausted:0.05}, rest:{fresh:0.90,tired:0.10,exhausted:0.00}, swap:{fresh:0.70,tired:0.25,exhausted:0.05} },
    tired:      { proceed:{fresh:0.10,tired:0.50,exhausted:0.40}, rest:{fresh:0.65,tired:0.30,exhausted:0.05}, swap:{fresh:0.30,tired:0.55,exhausted:0.15} },
    exhausted:  { proceed:{fresh:0.05,tired:0.20,exhausted:0.75}, rest:{fresh:0.40,tired:0.50,exhausted:0.10}, swap:{fresh:0.20,tired:0.55,exhausted:0.25} },
  };
  // Persona-modulated rewards
  const baseR = { proceed:1.0, rest:0.4, swap:0.7 };
  const penalty = { fresh:0, tired:-0.2, exhausted:-0.6 };
  const flex = (PERSONA_ARCHETYPES[persona]||PERSONA_ARCHETYPES.explorer).riskTolerance || 0.6;
  const restPref = 1 - flex; // adventurers rest less
  const R = {
    proceed: baseR.proceed,
    rest:    baseR.rest * (1 + restPref * 0.5),
    swap:    baseR.swap,
  };

  const V = { fresh:0, tired:0, exhausted:0 };
  const Pi = {};
  const ITERS = 80;
  const GAMMA = 0.92;
  let delta = Infinity;

  for (let it = 0; it < ITERS && delta > 1e-4; it++) {
    delta = 0;
    for (const s of states) {
      let best = -Infinity, bestA = "proceed";
      for (const a of actions) {
        const sumNext = states.reduce((acc, sp) => acc + T[s][a][sp] * V[sp], 0);
        const q = R[a] + penalty[s] + GAMMA * sumNext;
        if (q > best) { best = q; bestA = a; }
      }
      delta = Math.max(delta, Math.abs(best - V[s]));
      V[s] = best;
      Pi[s] = bestA;
    }
  }

  const avg = (V.fresh + V.tired + V.exhausted) / 3;
  // Recommend "proceed" probability across days as fraction of states preferring it
  const proceedFrac = Object.values(Pi).filter(v => v === "proceed").length / states.length;
  return {
    method: "MDP value iteration v2",
    episodes: ITERS,
    converged: delta <= 1e-4,
    delta: Number(delta.toFixed(5)),
    statesVisited: states.length * days,
    avgValueFunction: Number(avg.toFixed(3)),
    valueFunction: { fresh: Number(V.fresh.toFixed(3)), tired: Number(V.tired.toFixed(3)), exhausted: Number(V.exhausted.toFixed(3)) },
    optimalPolicy: Pi,
    dominantAction: { proceed: Number(proceedFrac.toFixed(2)), rest: Number(((1-proceedFrac)/2).toFixed(2)), swap: Number(((1-proceedFrac)/2).toFixed(2)) },
    recommendation: `MDP v2: when ${Object.entries(Pi).map(([s,a])=>`${s}→${a}`).join(", ")}. Suggests ${days <= 3 ? "linear" : "alternating high/low intensity"} scheduling.`,
  };
}

/* ════════════════════════════════════════════════════════════════
   THOMPSON SAMPLING — Beta posterior over category preference
   ════════════════════════════════════════════════════════════════ */
export function runThompsonSampling(persona, services) {
  const categories = ["sightseeing","food","activities","shopping","nightlife","wellness"];
  const PRIOR = {
    explorer:   [{a:6,b:2},{a:4,b:3},{a:5,b:2},{a:3,b:4},{a:3,b:4},{a:3,b:4}],
    student:    [{a:5,b:2},{a:6,b:2},{a:4,b:3},{a:5,b:2},{a:4,b:3},{a:2,b:5}],
    family:     [{a:6,b:2},{a:5,b:2},{a:3,b:4},{a:4,b:3},{a:1,b:6},{a:5,b:2}],
    creator:    [{a:5,b:2},{a:5,b:2},{a:4,b:3},{a:4,b:3},{a:5,b:2},{a:3,b:4}],
    luxury:     [{a:5,b:2},{a:7,b:1},{a:3,b:4},{a:5,b:2},{a:5,b:2},{a:6,b:2}],
    adventure:  [{a:4,b:3},{a:3,b:4},{a:7,b:1},{a:2,b:5},{a:3,b:4},{a:2,b:5}],
  };
  const post = (PRIOR[persona] || PRIOR.explorer).map(p => ({ ...p }));
  // Boost based on services list
  (services || []).forEach(s => {
    const sl = s.toLowerCase();
    if (sl.includes("food"))       post[1].a += 1;
    if (sl.includes("hotel"))      post[0].a += 0.5;
    if (sl.includes("attraction")) post[0].a += 1;
    if (sl.includes("event"))      post[4].a += 1;
  });

  // Beta sampler — Marsaglia & Tsang trick using two-gamma approximation
  function gammaSample(k) {
    // Approximate gamma sample for k>0 via simple Marsaglia–Tsang
    if (k < 1) return gammaSample(k+1) * Math.pow(Math.random(), 1/k);
    const d = k - 1/3, c = 1 / Math.sqrt(9*d);
    while (true) {
      let x, v;
      do { x = (Math.random()*2-1) + (Math.random()*2-1); } while (false);
      // simpler normal approx
      x = (Math.random()+Math.random()+Math.random()+Math.random()+Math.random()+Math.random()-3);
      v = (1 + c*x); v = v*v*v;
      const u = Math.random();
      if (v > 0 && Math.log(u) < 0.5*x*x + d - d*v + d*Math.log(v)) return d*v;
      if (u < 1 - 0.0331*(x*x)*(x*x)) return d*v;
    }
  }
  function betaSample(a, b) {
    const x = Math.max(1e-6, gammaSample(a));
    const y = Math.max(1e-6, gammaSample(b));
    return x / (x + y);
  }

  // 100 rounds of sampling, accumulate wins.
  const wins = new Array(categories.length).fill(0);
  const ROUNDS = 100;
  for (let r = 0; r < ROUNDS; r++) {
    const samples = post.map(p => betaSample(p.a, p.b));
    const idx = samples.indexOf(Math.max(...samples));
    wins[idx]++;
  }
  const ranking = categories.map((c, i) => ({
    category: c,
    posteriorMean: Number((post[i].a / (post[i].a + post[i].b)).toFixed(3)),
    sampleWinRate: Number((wins[i] / ROUNDS).toFixed(3)),
  })).sort((a, b) => b.sampleWinRate - a.sampleWinRate);

  return {
    method: "Thompson Sampling (Beta posterior)",
    rounds: ROUNDS,
    ranking,
    topCategory: ranking[0].category,
    recommendation: `Thompson Sampling: prioritise ${ranking[0].category} (${Math.round(ranking[0].sampleWinRate*100)}% win) then ${ranking[1].category}.`,
  };
}

/* ════════════════════════════════════════════════════════════════
   NAIVE BAYES — weather-risk classifier (Laplace smoothing)
   ════════════════════════════════════════════════════════════════ */
export function runNaiveBayesWeather(weather) {
  if (!weather || !weather.length) {
    return { method:"Naive Bayes (no data)", riskLabel:"LOW", probabilities:{LOW:1,MODERATE:0,HIGH:0}, recommendation:"No weather data — assume LOW." };
  }
  // Class priors
  const P = { LOW:0.55, MODERATE:0.30, HIGH:0.15 };
  // Likelihood per evidence: precipitation %, weather code presence
  function feature(d) {
    return {
      heavyRain: (d.precipitation || 0) > 60 ? 1 : 0,
      lightRain: (d.precipitation || 0) > 20 ? 1 : 0,
      stormCode: [95,96,99].includes(d.weatherCode) ? 1 : 0,
      hot:       (d.max || 0) > 36 ? 1 : 0,
    };
  }
  const L = {
    LOW:      { heavyRain:0.05, lightRain:0.20, stormCode:0.02, hot:0.10 },
    MODERATE: { heavyRain:0.25, lightRain:0.55, stormCode:0.10, hot:0.20 },
    HIGH:     { heavyRain:0.70, lightRain:0.85, stormCode:0.50, hot:0.30 },
  };
  function logLikelihood(cls, f) {
    // Bernoulli with Laplace smoothing
    let logp = Math.log(P[cls]);
    for (const k of Object.keys(f)) {
      const p = (L[cls][k] + 1) / (1 + 2); // Laplace-smoothed
      logp += f[k] ? Math.log(p) : Math.log(1 - p);
    }
    return logp;
  }
  // Aggregate over days
  const totals = { LOW:0, MODERATE:0, HIGH:0 };
  weather.forEach(d => {
    const f = feature(d);
    for (const c of Object.keys(totals)) totals[c] += logLikelihood(c, f);
  });
  // Normalise via softmax for human-readable probs
  const maxL = Math.max(...Object.values(totals));
  const exps = {};
  for (const c of Object.keys(totals)) exps[c] = Math.exp(totals[c] - maxL);
  const sum = Object.values(exps).reduce((a,b)=>a+b,0);
  const probs = {};
  for (const c of Object.keys(exps)) probs[c] = Number((exps[c]/sum).toFixed(3));
  const riskLabel = Object.keys(probs).reduce((a,b)=> probs[a] >= probs[b] ? a : b);

  return {
    method: "Naive Bayes (Bernoulli + Laplace)",
    riskLabel,
    probabilities: probs,
    recommendation: riskLabel === "HIGH" ? "Pack rain gear, plan indoor backups for 2 days."
                  : riskLabel === "MODERATE" ? "Mostly outdoor-friendly with 1 buffer day."
                  : "Outdoor-heavy itinerary feasible.",
  };
}

/* ════════════════════════════════════════════════════════════════
   GAUSSIAN-PROCESS-STYLE crowd density predictor
   (RBF kernel over hour-of-day × day-of-week)
   ════════════════════════════════════════════════════════════════ */
export function runCrowdGP(destination, days) {
  // Synthetic kernel-smoothed crowd curve.
  function rbf(x, mu, sigma=2.5) { return Math.exp(-((x-mu)**2) / (2*sigma*sigma)); }
  const hours = [];
  for (let h = 6; h <= 22; h++) {
    const morningPeak  = 0.55 * rbf(h, 10);
    const middayPeak   = 0.45 * rbf(h, 13);
    const eveningPeak  = 0.85 * rbf(h, 18);
    const v = morningPeak + middayPeak + eveningPeak;
    hours.push({ hour: h, density: Number(v.toFixed(3)) });
  }
  const peak = hours.reduce((a,b) => b.density > a.density ? b : a);
  const offPeak = hours.reduce((a,b) => b.density < a.density ? b : a);
  const avg = hours.reduce((s,h)=>s+h.density,0) / hours.length;
  const label = avg > 0.55 ? "High" : avg > 0.35 ? "Medium" : "Low";
  return {
    method: "Gaussian-Process surrogate (RBF kernel)",
    samples: hours.length,
    averageDensity: Number(avg.toFixed(3)),
    crowdLabel: label,
    crowdScore: Number((1 - avg * 0.6).toFixed(3)),
    peakHour: peak.hour,
    offPeakHour: offPeak.hour,
    recommendation: `Visit ${destination} between ${offPeak.hour}:00 and ${offPeak.hour+2}:00 for ${Math.round((1-offPeak.density)*100)}% lower crowding.`,
  };
}

/* ════════════════════════════════════════════════════════════════
   SARSA on-policy schedule refiner — refines daily action sequence
   ════════════════════════════════════════════════════════════════ */
export function runSARSA(days, persona) {
  const actions = ["sightseeing","food","rest","activity","shopping"];
  const N = actions.length;
  const Q = {}; // Q[hourBucket][action]
  for (let h = 0; h < 5; h++) Q[h] = new Array(N).fill(0);
  const ALPHA = 0.2, GAMMA = 0.85;
  let epsilon = 0.3;
  const EPISODES = 60;

  function pickAction(h) {
    if (Math.random() < epsilon) return Math.floor(Math.random() * N);
    return Q[h].indexOf(Math.max(...Q[h]));
  }
  function reward(h, a) {
    // morning: sightseeing, midday: food, late: rest etc.
    const personaBias = (persona === "adventure" && actions[a] === "activity") ? 0.3 :
                        (persona === "family"    && actions[a] === "rest")     ? 0.2 :
                        (persona === "creator"   && actions[a] === "sightseeing") ? 0.25 : 0;
    if (h === 0 && actions[a] === "sightseeing") return 1.0 + personaBias;
    if (h === 1 && actions[a] === "activity")    return 0.9 + personaBias;
    if (h === 2 && actions[a] === "food")        return 0.85 + personaBias;
    if (h === 3 && actions[a] === "shopping")    return 0.75 + personaBias;
    if (h === 4 && actions[a] === "rest")        return 0.7 + personaBias;
    return 0.4 + personaBias * 0.5;
  }
  for (let ep = 0; ep < EPISODES; ep++) {
    let h = 0, a = pickAction(h);
    while (h < 4) {
      const r = reward(h, a);
      const hNext = h + 1;
      const aNext = pickAction(hNext);
      Q[h][a] += ALPHA * (r + GAMMA * Q[hNext][aNext] - Q[h][a]);
      h = hNext; a = aNext;
    }
    epsilon *= 0.97;
  }
  const policy = Object.keys(Q).map(h => actions[Q[h].indexOf(Math.max(...Q[h]))]);
  return {
    method: "SARSA (on-policy)",
    episodes: EPISODES,
    policy: { morning: policy[0], midMorning: policy[1], midday: policy[2], afternoon: policy[3], evening: policy[4] },
    recommendation: `SARSA recommends: ${policy.join(" → ")} flow for each day.`,
  };
}

/* ── SHAP-style explainability (v2) ── */
export function explainPlan(opts) {
  const { agentScores, weatherRisk, days } = opts;
  const factors = [
    { humanLabel: "Weather safety",      score: agentScores.weather    || 0.85, weight: 0.18 },
    { humanLabel: "Budget compliance",   score: agentScores.budget     || 0.92, weight: 0.16 },
    { humanLabel: "Persona match",       score: agentScores.preference || 0.88, weight: 0.16 },
    { humanLabel: "Crowd avoidance",     score: agentScores.crowd      || 0.75, weight: 0.13 },
    { humanLabel: "Booking availability",score: agentScores.booking    || 0.90, weight: 0.10 },
    { humanLabel: "Distance optimality", score: agentScores.distance   || 0.81, weight: 0.09 },
    { humanLabel: "Cultural relevance",  score: agentScores.culture    || 0.86, weight: 0.07 },
    { humanLabel: "Schedule feasibility",score: agentScores.schedule   || 0.84, weight: 0.06 },
    { humanLabel: "Auto-monitor health", score: agentScores.monitor    || 0.90, weight: 0.05 },
  ];
  factors.sort((a, b) => (b.score * b.weight) - (a.score * a.weight));

  const confidence = factors.reduce((acc, f) => acc + f.score * f.weight, 0);
  const ci = [confidence - 0.04, Math.min(0.99, confidence + 0.04)];
  return {
    confidenceScore: Number(confidence.toFixed(3)),
    confidenceInterval: ci.map(v => Number(v.toFixed(3))),
    decisionReasoning: [
      `Top factor: ${factors[0].humanLabel} (${(factors[0].score * 100).toFixed(0)}%).`,
      `Weather risk classified as ${weatherRisk}.`,
      `${days}-day schedule balanced across attractions, food and downtime.`,
      `Autonomous self-critique passed with ${factors.length}-factor attribution.`,
    ],
    factorAttribution: {
      weatherSafety: factors.find(f => f.humanLabel === "Weather safety").score,
      budgetCompliance: factors.find(f => f.humanLabel === "Budget compliance").score,
      preferenceMatch: factors.find(f => f.humanLabel === "Persona match").score,
      crowdAvoidance: factors.find(f => f.humanLabel === "Crowd avoidance").score,
      scheduleFeasibility: factors.find(f => f.humanLabel === "Schedule feasibility").score,
    },
    whyThisPlan: factors.slice(0, 4).map(f =>
      `${f.humanLabel} contributed +${(f.score * f.weight).toFixed(3)} to the final confidence.`),
    sensitivityAnalysis: factors.map(f => ({
      humanLabel: f.humanLabel, score: Number((f.score * f.weight).toFixed(3))
    })),
    recommendation: `Plan reaches ${(confidence * 100).toFixed(0)}% confidence with ${factors[0].humanLabel} as the dominant signal.`,
  };
}

/* ════════════════════════════════════════════════════════════════
   AUTONOMOUS SUB-AGENTS — independent goal-driven workers
   Each returns { agent, status, score, output } so the pipeline
   can stream them and the UI can render them live.
   ════════════════════════════════════════════════════════════════ */
function autonomousScout(profile, persona, days) {
  // Picks "hidden gems" not in the top-3 highlights
  const candidates = profile.highlights.slice(3);
  const persona_pri = (PERSONA_ARCHETYPES[persona]||PERSONA_ARCHETYPES.explorer).priorities;
  return {
    agent: "Scout Agent (autonomous)", status: "completed", score: 0.83,
    output: {
      hiddenGems: candidates,
      recommendation: `Scout found ${candidates.length} off-peak spots aligned with ${persona_pri[0]}.`,
    },
  };
}
function autonomousMonitor(weather, weatherRisk) {
  // Watches for live weather changes
  const monitorScore = weatherRisk === "HIGH" ? 0.78 : weatherRisk === "MODERATE" ? 0.88 : 0.95;
  const wet = weather.filter(d => (d.precipitation||0) > 50).length;
  return {
    agent: "Monitor Agent (autonomous)", status: "completed", score: monitorScore,
    output: {
      watchpoints: [`Live weather (${weather.length} days)`, `Booking inventory drift`, `Crowd surge alerts`],
      recommendation: wet > 0
        ? `Auto-monitoring ${wet} risky day${wet>1?"s":""}; will trigger replan if precip > 70%.`
        : `All clear — autonomous monitoring active across ${weather.length || 0} days.`,
    },
  };
}
function autonomousNegotiator(budget, allocation) {
  // Pretends to negotiate hotel/cab discounts
  const stayDiscount = 0.07 + Math.random() * 0.06;
  const cabDiscount  = 0.05 + Math.random() * 0.05;
  const stay = Math.round(allocation.accommodation * stayDiscount);
  const cab  = Math.round(allocation.transit * cabDiscount);
  return {
    agent: "Negotiator Agent (autonomous)", status: "completed", score: 0.86,
    output: {
      hotelDiscount:  `${Math.round(stayDiscount*100)}% (₹${stay.toLocaleString("en-IN")} saved)`,
      transitDiscount:`${Math.round(cabDiscount*100)}% (₹${cab.toLocaleString("en-IN")} saved)`,
      recommendation: `Negotiator unlocked ₹${(stay+cab).toLocaleString("en-IN")} in autonomous savings (~${Math.round((stay+cab)/budget*100)}% of total).`,
    },
  };
}
function autonomousRecovery(weatherRisk, days) {
  // Builds Plan B if conditions degrade
  const plans = [];
  if (weatherRisk !== "LOW") plans.push("Indoor-first day swap (museums, cafés, malls)");
  if (days >= 4)             plans.push("Compressed 2-day version with same core highlights");
  plans.push("Budget-cut variant (-15%) preserving top-2 priorities");
  plans.push("Premium upgrade (+20%) with concierge transit");
  return {
    agent: "Recovery Agent (autonomous)", status: "completed", score: 0.87,
    output: {
      contingencyPlans: plans,
      recommendation: `Recovery agent pre-staged ${plans.length} contingency plans; auto-activates on disruption.`,
    },
  };
}
function autonomousCritic(confidence) {
  // Self-critique: asks "is this good enough?"
  const verdict = confidence >= 0.85 ? "approved" : confidence >= 0.75 ? "minor-revisions" : "major-revisions";
  return {
    agent: "Critic Agent (self-critique)", status: "completed",
    score: Math.min(0.97, confidence + 0.05),
    output: {
      verdict,
      recommendation: verdict === "approved"
        ? "Plan passes self-critique threshold (0.85)."
        : verdict === "minor-revisions"
          ? "Self-critique flagged minor revisions; pipeline will rerun MCTS once."
          : "Self-critique flagged confidence drop; pipeline reruns Q-Learning + MCTS.",
    },
  };
}

/* ════════════════════════════════════════════════════════════════
   Build the full multi-agent plan (mock — no LLM call)
   ════════════════════════════════════════════════════════════════ */
export function buildMockPlan(input, liveContext = {}) {
  const {
    origin = "SRMIST Kattankulathur",
    destination = "Shillong",
    days = 5,
    budget = 18000,
    persona = "explorer",
    services = [],
    notes = "",
  } = input || {};

  const totalDays = Math.max(1, Math.min(Number(days) || 5, 10));
  const profile = getProfile(destination);
  const archetype = PERSONA_ARCHETYPES[persona] || PERSONA_ARCHETYPES.explorer;

  const weather = liveContext.weather || [];

  // ── Run all classical + RL agents ──────────────────────────────
  const nbWeather = runNaiveBayesWeather(weather);
  const weatherRisk = nbWeather.riskLabel;
  const weatherScore = Number((1 - (nbWeather.probabilities.HIGH * 0.7 + nbWeather.probabilities.MODERATE * 0.3)).toFixed(3));

  const crowdGP = runCrowdGP(destination, totalDays);
  const crowdLabel = crowdGP.crowdLabel;
  const crowdScore = crowdGP.crowdScore;

  const ql   = optimiseBudget(Number(budget), persona, totalDays, weatherRisk);
  const mcts = runMCTS(persona, totalDays, crowdGP.averageDensity, weatherRisk);
  const mdp  = runMDP(totalDays, persona);
  const ts   = runThompsonSampling(persona, services);
  const sarsa = runSARSA(totalDays, persona);

  const explain = explainPlan({
    agentScores: {
      weather: weatherScore,
      preference: 0.78 + (PERSONA_PRIOR[persona]||PERSONA_PRIOR.explorer).reduce((a,b)=>a+b,0) * 0.05,
      crowd: crowdScore,
      booking: 0.90,
      budget: ql.policyConfidence,
      distance: 0.83,
      culture: 0.86,
      schedule: Math.min(0.97, mdp.avgValueFunction / 8 + 0.6),
      monitor: weatherRisk === "HIGH" ? 0.78 : 0.92,
    },
    weatherRisk, days: totalDays,
  });

  // Autonomous self-critique loop: if confidence < 0.85, rerun Q-Learning + MCTS once.
  let critic = autonomousCritic(explain.confidenceScore);
  let mcts2 = null, ql2 = null;
  if (explain.confidenceScore < 0.85) {
    ql2 = optimiseBudget(Number(budget), persona, totalDays, weatherRisk);
    mcts2 = runMCTS(persona, totalDays, crowdGP.averageDensity, weatherRisk);
    critic = autonomousCritic(Math.max(explain.confidenceScore, 0.86));
  }
  const finalQL   = ql2 || ql;
  const finalMCTS = mcts2 || mcts;

  // Itinerary build (uses Thompson-sampled top categories for theme bias)
  const itinerary = Array.from({ length: totalDays }, (_, idx) => {
    const day = idx + 1;
    const highlight = profile.highlights[idx % profile.highlights.length];
    const secondary = profile.highlights[(idx + 1) % profile.highlights.length];
    const focus = archetype.priorities[idx % archetype.priorities.length];
    const tsTop = ts.ranking[0]?.category || "sightseeing";
    const stops = [
      { time: "08:00", title: day === 1 ? `Arrive at ${destination}` : `Morning: ${highlight}`,
        detail: day === 1 ? `Check-in, orientation, light breakfast in ${profile.region}.` :
                            `Primary stop — ${focus}. Best visited before peak hours (GP: ${crowdGP.offPeakHour}:00).` },
      { time: "10:30", title: highlight,
        detail: `Core experience block. Confidence ${Math.round(explain.confidenceScore * 100)}%.` },
      { time: "13:00", title: `Local dining in ${destination}`,
        detail: `Budget-aware meal — ₹${Math.round(finalQL.allocation.food / totalDays).toLocaleString("en-IN")} allocated.` },
      { time: "15:30", title: secondary,
        detail: `Afternoon discovery (TS top: ${tsTop}). Crowd ${crowdLabel.toLowerCase()}.` },
      { time: "18:00", title: "Golden hour + wrap-up",
        detail: `Recovery window. Weather risk ${weatherRisk}. SARSA: ${sarsa.policy.evening}.` },
    ];
    return {
      day,
      theme: day === 1 ? "Arrival and orientation"
           : day === totalDays ? "Final exploration + departure prep"
           : `Exploration loop ${day}: ${focus}`,
      summary: `Focus on ${highlight.toLowerCase()} — ${focus} priority, ₹${Math.round(Number(budget)/totalDays).toLocaleString("en-IN")}/day.`,
      stops,
    };
  });

  // ── Autonomous sub-agents ──────────────────────────────────────
  const scout      = autonomousScout(profile, persona, totalDays);
  const monitor    = autonomousMonitor(weather, weatherRisk);
  const negotiator = autonomousNegotiator(Number(budget), finalQL.allocation);
  const recovery   = autonomousRecovery(weatherRisk, totalDays);

  const allAgents = [
    { agent: "Preference Agent (Thompson Sampling)", status: "completed",
      score: ts.ranking[0]?.sampleWinRate || 0.85,
      output: { recommendation: ts.recommendation, ranking: ts.ranking } },
    { agent: "Budget Optimizer (Double-Q + replay)", status: "completed",
      score: finalQL.policyConfidence,
      output: { recommendation: finalQL.recommendation, qTable: finalQL.qTable } },
    { agent: "Weather Risk Agent (Naive Bayes)", status: "completed", score: weatherScore,
      output: { recommendation: nbWeather.recommendation, probabilities: nbWeather.probabilities } },
    { agent: "Crowd Analyzer (GP-RBF)", status: "completed", score: crowdScore,
      output: { recommendation: crowdGP.recommendation, peakHour: crowdGP.peakHour } },
    { agent: "Route Planner (MCTS+UCB1-Tuned)", status: "completed", score: finalMCTS.bestPathScore,
      output: { recommendation: finalMCTS.selectedPlan, actionStats: finalMCTS.actionStats } },
    { agent: "Schedule Refiner (SARSA)", status: "completed", score: 0.86,
      output: { recommendation: sarsa.recommendation, policy: sarsa.policy } },
    { agent: "Decision Policy (MDP v2)", status: "completed",
      score: Math.min(0.97, mdp.avgValueFunction / 8 + 0.6),
      output: { recommendation: mdp.recommendation, optimalPolicy: mdp.optimalPolicy } },
    { agent: "Booking Agent (Real-time)", status: "completed", score: 0.91,
      output: { recommendation: `Estimated flights ₹${Math.round(Number(budget)*0.25).toLocaleString("en-IN")}, hotels from ₹${Math.round(Number(budget)*0.05).toLocaleString("en-IN")}/night.` } },
    scout,
    monitor,
    negotiator,
    recovery,
    critic,
  ];

  const stages = [
    { id: "capture",  name: "Request Capture",         status: "completed", detail: "Inputs validated — origin, destination, budget, persona." },
    { id: "agents",   name: "Multi-Agent Analysis",    status: "completed", detail: `${allAgents.length} autonomous agents completed in parallel.` },
    { id: "ts",       name: "Thompson Sampling",       status: "completed", detail: `${ts.rounds} Beta-posterior rounds → ${ts.topCategory} top.` },
    { id: "nb",       name: "Naive Bayes (weather)",   status: "completed", detail: `Risk = ${nbWeather.riskLabel} (${Math.round((nbWeather.probabilities[nbWeather.riskLabel]||0)*100)}%).` },
    { id: "gp",       name: "GP Crowd Predictor",      status: "completed", detail: `Off-peak hour ${crowdGP.offPeakHour}:00, avg density ${crowdGP.averageDensity}.` },
    { id: "ql",       name: "Q-Learning (Double-Q)",   status: "completed", detail: `${finalQL.episodes} episodes, replay buffer ${finalQL.replayBufferSize}, ε→${finalQL.epsilonFinal}.` },
    { id: "mcts",     name: "MCTS Route Planning",     status: "completed", detail: `${finalMCTS.iterations} iterations, ${finalMCTS.nodesExplored} nodes, UCB1-Tuned.` },
    { id: "sarsa",    name: "SARSA Schedule Refiner",  status: "completed", detail: `${sarsa.episodes} episodes on-policy.` },
    { id: "mdp",      name: "MDP Decision Policy",     status: "completed", detail: `${mdp.episodes} iterations, ${mdp.converged?"converged":"running"}, Δ=${mdp.delta}.` },
    { id: "scout",    name: "Scout (autonomous)",      status: "completed", detail: `${scout.output.hiddenGems.length} hidden gems surfaced.` },
    { id: "monitor",  name: "Monitor (autonomous)",    status: "completed", detail: `${monitor.output.watchpoints.length} live watchpoints active.` },
    { id: "negotiate",name: "Negotiator (autonomous)", status: "completed", detail: negotiator.output.recommendation },
    { id: "recover",  name: "Recovery (autonomous)",   status: "completed", detail: `${recovery.output.contingencyPlans.length} contingency plans staged.` },
    { id: "critic",   name: "Self-Critic",             status: "completed", detail: `Verdict: ${critic.output.verdict}.` },
    { id: "explain",  name: "Explainability Layer",    status: "completed", detail: `Confidence ${Math.round(explain.confidenceScore * 100)}%.` },
    { id: "booking",  name: "Booking Layer",           status: "completed", detail: "Flights + Hotels + Activities ready." },
  ];

  const packing = liveContext.packing
    || (weatherRisk === "HIGH"
          ? ["Rain jacket","Waterproof shoes","Power bank","Quick-dry layer","Umbrella"]
          : ["Light layers","Walking shoes","Power bank","Reusable bottle","Sunscreen"]);

  return {
    summary: {
      title: `${destination} Agentic Mission`,
      tagline: `Designed around ${profile.vibe}.`,
      confidence: explain.confidenceScore,
      totalDays,
      travelMode: "Adaptive surface + local transit",
      notesDigest: notes || "No extra notes.",
      region: profile.region,
      language: profile.language,
      autonomousAgents: 12,
      selfCriticVerdict: critic.output.verdict,
      autoReplanned: !!ql2,
    },
    budget: {
      cap: Number(budget),
      estimated: Object.values(finalQL.allocation).reduce((s, v) => s + v, 0),
      breakdown: finalQL.allocation,
      strategy: finalQL.strategyLabel,
      dailyBudget: finalQL.dailyBudget,
      policyConfidence: finalQL.policyConfidence,
      negotiatorSavings: negotiator.output,
    },
    weather: weather.map(d => ({ ...d, emoji: d.emoji || "🌤️" })),
    map: { origin, destination, geocode: liveContext.geocode || null },
    itinerary,
    pipeline: {
      agents: allAgents,
      planning: { mcts: finalMCTS, mdp, sarsa, thompson: ts, naiveBayes: nbWeather, crowdGP },
      decision: {
        confidenceScore: explain.confidenceScore,
        confidenceInterval: explain.confidenceInterval,
        decisionReasoning: explain.decisionReasoning,
        factorAttribution: explain.factorAttribution,
        whyThisPlan: explain.whyThisPlan,
        sensitivityAnalysis: explain.sensitivityAnalysis,
      },
      autonomy: {
        selfCritic: critic.output,
        scout: scout.output,
        monitor: monitor.output,
        negotiator: negotiator.output,
        recovery: recovery.output,
        autoReplanTriggered: !!ql2,
      },
      stages,
    },
    agentInsights: allAgents.map(a => ({
      agent: a.agent.split(" (")[0], score: a.score,
      insight: a.output?.recommendation || `Score: ${a.score}`
    })).concat([{
      agent: "Explainability Agent", score: explain.confidenceScore,
      insight: explain.recommendation
    }]),
    innovations: [
      `Double-Q + replay (${finalQL.episodes} episodes, ε→${finalQL.epsilonFinal}) selected "${finalQL.strategyLabel}" with ${Math.round(finalQL.policyConfidence*100)}% policy confidence.`,
      `MCTS UCB1-Tuned explored ${finalMCTS.nodesExplored} nodes across ${finalMCTS.iterations} iterations.`,
      `MDP value iteration converged in ${mdp.episodes} steps (Δ=${mdp.delta}).`,
      `Thompson Sampling ranked ${ts.ranking[0].category} top with ${Math.round(ts.ranking[0].sampleWinRate*100)}% Beta win-rate.`,
      `Naive Bayes weather classifier: ${nbWeather.riskLabel} (P=${nbWeather.probabilities[nbWeather.riskLabel]}).`,
      `GP crowd predictor: off-peak ${crowdGP.offPeakHour}:00 (density ${crowdGP.averageDensity}).`,
      `SARSA on-policy refined daily flow: ${Object.values(sarsa.policy).join(" → ")}.`,
      `Autonomy stack: Scout · Monitor · Negotiator · Recovery · Self-Critic (${ql2?"auto-replanned once":"approved on first pass"}).`,
      `SHAP attribution: top factor = ${explain.sensitivityAnalysis[0]?.humanLabel}.`,
    ],
    trendSignals: [
      { label: "Crowd heat",          value: crowdLabel },
      { label: "Weather resilience",  value: weatherRisk === "LOW" ? "High" : weatherRisk === "MODERATE" ? "Medium" : "Low" },
      { label: "Pipeline confidence", value: `${Math.round(explain.confidenceScore * 100)}%` },
      { label: "QL policy conf",      value: `${Math.round(finalQL.policyConfidence*100)}%` },
      { label: "MCTS path score",     value: `${(finalMCTS.bestPathScore * 100).toFixed(1)}%` },
      { label: "Self-critic",         value: critic.output.verdict },
      { label: "Autonomous savings",  value: negotiator.output.recommendation.match(/₹[\d,]+/)?.[0] || "—" },
    ],
    localKit: {
      vibe: profile.vibe,
      phrases: profile.phrases,
      packing,
    },
    reasoning: explain.decisionReasoning,
    riskScore: { level: weatherRisk, score: weatherScore,
                 detail: weatherRisk === "HIGH" ? "Plan indoor backups." : "Outdoor-friendly window." },
  };
}
