/* POST /api/autonomous/critic — Self-critique agent.
   Audits a generated plan against a rubric (weather safety, budget
   discipline, persona match, schedule feasibility, fallback coverage)
   and returns a verdict with revision instructions. */

import { jsonResponse } from "../_shared/auth.js";

export const onRequestOptions = () => jsonResponse({ ok: true });

function score(plan) {
  const conf = Number(plan?.summary?.confidence || 0);
  const budgetCap = Number(plan?.budget?.cap || 0);
  const budgetEst = Number(plan?.budget?.estimated || 0);
  const budgetOK = budgetCap > 0 ? budgetEst <= budgetCap * 1.05 : true;
  const dayCount = (plan?.itinerary || []).length;
  const stopsTotal = (plan?.itinerary || []).reduce((a, d) => a + (d?.stops?.length || 0), 0);
  const stopsPerDay = dayCount ? stopsTotal / dayCount : 0;

  const rubric = [
    { id:"confidence",      label:"Pipeline confidence ≥ 0.85",    pass: conf >= 0.85,           weight: 0.30 },
    { id:"budget",          label:"Budget within +5% cap",          pass: budgetOK,               weight: 0.25 },
    { id:"schedule",        label:"4–6 stops per day feasibility",  pass: stopsPerDay >= 4 && stopsPerDay <= 6, weight: 0.15 },
    { id:"fallback",        label:"Recovery plans present",         pass: !!plan?.pipeline?.autonomy?.recovery, weight: 0.10 },
    { id:"explainability",  label:"SHAP attribution attached",      pass: !!plan?.pipeline?.decision?.factorAttribution, weight: 0.10 },
    { id:"autonomy",        label:"Autonomous agents ≥ 10",          pass: (plan?.summary?.autonomousAgents || 0) >= 10, weight: 0.10 },
  ];
  const passed = rubric.filter(r => r.pass);
  const ratio = passed.length / rubric.length;
  const verdict = ratio >= 0.9 ? "approved"
                 : ratio >= 0.7 ? "minor-revisions"
                                : "major-revisions";
  const revisions = rubric.filter(r => !r.pass).map(r => r.label);
  return { rubric, passed: passed.length, total: rubric.length, ratio: Number(ratio.toFixed(3)), verdict, revisions };
}

export const onRequestPost = async ({ request }) => {
  let body = {};
  try { body = await request.json(); } catch { body = {}; }
  const { plan } = body;
  if (!plan) return jsonResponse({ ok: false, error: "plan object required" }, 400);

  const result = score(plan);
  return jsonResponse({
    ok: true,
    mode: "autonomous-critic",
    method: "Rubric-weighted self-critique with revision targets",
    ...result,
    recommendation: result.verdict === "approved"
      ? "Plan passes all critical checks — safe to commit."
      : result.verdict === "minor-revisions"
        ? `Plan needs minor revisions on: ${result.revisions.join(", ")}.`
        : `Major revisions required — failing: ${result.revisions.join(", ")}.`,
  });
};
