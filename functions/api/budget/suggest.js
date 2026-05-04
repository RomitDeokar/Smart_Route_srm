/* GET /api/budget/suggest — AI suggestions for current budget. */

import { jsonResponse } from "../_shared/auth.js";
import { _budgetGet } from "./_store.js";

export const onRequestOptions = () => jsonResponse({ ok: true });

export const onRequestGet = () => {
  const budget = _budgetGet();
  if (!budget) {
    return jsonResponse({ ok: true,
      suggestions: [{ type: "info", text: "Create a budget first." }],
      health: "unknown", overallSpentPct: 0
    });
  }

  const suggestions = [];
  Object.entries(budget.categories || {}).forEach(([key, val]) => {
    if (val.progress > 85) suggestions.push({ type: "danger",
      text: `${key} critically low — ${val.progress}% used!` });
    else if (val.progress > 60) suggestions.push({ type: "warning",
      text: `${key} at ${val.progress}% — monitor closely.` });
  });

  const { Hotels, Shopping, Food, Activities } = budget.categories || {};
  if (Hotels?.progress > 75 && Shopping?.remaining > 2000) {
    suggestions.push({ type: "suggestion",
      text: "Move ₹2,000 from Shopping → Hotels for a better stay." });
  }
  if (Activities?.progress > 80 && Food?.remaining > 1500) {
    suggestions.push({ type: "suggestion",
      text: "Move ₹1,500 from Food → Activities for more experiences." });
  }
  if (!suggestions.length) suggestions.push({ type: "success",
    text: "Budget allocation looks healthy — all categories within safe limits." });

  const totalPct = budget.totalBudget > 0
    ? Math.round((budget.totalSpent / budget.totalBudget) * 100) : 0;

  return jsonResponse({ ok: true, suggestions,
    health: totalPct < 60 ? "excellent" : totalPct < 85 ? "moderate" : "critical",
    overallSpentPct: totalPct });
};
