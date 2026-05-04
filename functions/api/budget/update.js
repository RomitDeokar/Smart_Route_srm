/* POST /api/budget/update — log an expense in a category. */

import { jsonResponse } from "../_shared/auth.js";
import { _budgetGet, _budgetSet, _formatBudget, BUDGET_CATS } from "./_store.js";

export const onRequestOptions = () => jsonResponse({ ok: true });

export const onRequestPost = async ({ request }) => {
  let body = {};
  try { body = await request.json(); } catch { body = {}; }
  const { category, amount } = body || {};

  const current = _budgetGet();
  if (!current) return jsonResponse({ ok: false, error: "Create a budget first." }, 400);
  if (!BUDGET_CATS.includes(category))
    return jsonResponse({ ok: false, error: "Invalid category." }, 400);

  const spent = BUDGET_CATS.reduce((a, c) =>
    ({ ...a, [c]: current.categories[c].spent }), {});
  spent[category] += Number(amount || 0);
  const allocs = BUDGET_CATS.reduce((a, c) =>
    ({ ...a, [c]: current.categories[c].allocated }), {});

  const updated = _formatBudget(current.totalBudget, allocs, spent);
  _budgetSet(updated);
  return jsonResponse({ ok: true, budget: updated });
};
