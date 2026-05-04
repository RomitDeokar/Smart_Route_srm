/* POST /api/budget/create — initialize budget store (per-isolate). */

import { jsonResponse } from "../_shared/auth.js";
import { _budgetSet, _formatBudget, BUDGET_CATS, defaultAlloc } from "./_store.js";

export const onRequestOptions = () => jsonResponse({ ok: true });

export const onRequestPost = async ({ request }) => {
  let body = {};
  try { body = await request.json(); } catch { body = {}; }
  const { total_budget, totalBudget: alt, allocations } = body || {};
  const total = Number(total_budget || alt || 0);
  if (!total) return jsonResponse({ ok: false, error: "Total budget is required." }, 400);

  const mapped = { ...(allocations || {}) };
  if (mapped.Stay && !mapped.Hotels) { mapped.Hotels = mapped.Stay; delete mapped.Stay; }
  const norm = BUDGET_CATS.reduce((a, c) =>
    ({ ...a, [c]: Number(mapped?.[c] ?? defaultAlloc(total)[c]) }), {});

  const budget = _formatBudget(total, norm);
  _budgetSet(budget);
  return jsonResponse({ ok: true, budget });
};
