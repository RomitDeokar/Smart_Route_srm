/* Shared in-memory budget store for /api/budget/* endpoints. */

export const BUDGET_CATS = ["Food","Shopping","Transport","Hotels","Activities"];

export function defaultAlloc(total) {
  return {
    Food: Math.round(total * 0.20),
    Shopping: Math.round(total * 0.15),
    Transport: Math.round(total * 0.20),
    Hotels: Math.round(total * 0.30),
    Activities: Math.round(total * 0.15),
  };
}

export function _formatBudget(total, allocs, spent = {}) {
  const categories = BUDGET_CATS.reduce((acc, cat) => {
    const allocated = Number(allocs[cat] || 0);
    const s = Number(spent[cat] || 0);
    acc[cat] = {
      allocated, spent: s,
      remaining: allocated - s,
      progress: allocated > 0 ? Math.round((s / allocated) * 100) : 0
    };
    return acc;
  }, {});
  return {
    totalBudget: Number(total),
    totalSpent: Object.values(categories).reduce((s, c) => s + c.spent, 0),
    categories,
  };
}

let _budget = null;
export const _budgetGet = () => _budget;
export const _budgetSet = (b) => { _budget = b; };
