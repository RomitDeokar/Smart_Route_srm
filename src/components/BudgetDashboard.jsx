import { useEffect, useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import FeatureShell from "./FeatureShell.jsx";

const defaultCategories = [
  { id: "Stay", icon: "🏨", color: "#667eea" },
  { id: "Food", icon: "🍽️", color: "#10b981" },
  { id: "Transport", icon: "🚗", color: "#fb923c" },
  { id: "Activities", icon: "🎯", color: "#8b5cf6" },
  { id: "Shopping", icon: "🛍️", color: "#ec4899" }
];

function formatMoney(v) {
  return `₹${Number(v || 0).toLocaleString("en-IN")}`;
}

function getHealthColor(pct) {
  if (pct < 60) return "var(--green)";
  if (pct < 85) return "var(--gold)";
  return "var(--red)";
}

function getHealthLabel(pct) {
  if (pct < 60) return "Healthy";
  if (pct < 85) return "Warning";
  return "Exceeded";
}

function BudgetDashboard({
  initialTotalBudget,
  budgetStatus,
  loading,
  error,
  onCreateBudget,
  onAddExpense
}) {
  const [totalBudget, setTotalBudget] = useState(initialTotalBudget || 18000);
  const [categories] = useState(defaultCategories);
  const [allocations, setAllocations] = useState(() => {
    const total = initialTotalBudget || 18000;
    return {
      Stay: Math.round(total * 0.30),
      Food: Math.round(total * 0.20),
      Transport: Math.round(total * 0.20),
      Activities: Math.round(total * 0.15),
      Shopping: Math.round(total * 0.15)
    };
  });
  const [expenseCategory, setExpenseCategory] = useState("Food");
  const [expenseAmount, setExpenseAmount] = useState(500);
  const [showChart, setShowChart] = useState(false);

  useEffect(() => {
    setTotalBudget(initialTotalBudget || 18000);
  }, [initialTotalBudget]);

  const allocatedTotal = useMemo(
    () => Object.values(allocations).reduce((s, v) => s + v, 0),
    [allocations]
  );

  const remaining = totalBudget - allocatedTotal;

  const handleSliderChange = (catId, val) => {
    const numVal = Number(val);
    setAllocations(prev => ({ ...prev, [catId]: numVal }));
  };

  const resetToSuggested = () => {
    setAllocations({
      Stay: Math.round(totalBudget * 0.30),
      Food: Math.round(totalBudget * 0.20),
      Transport: Math.round(totalBudget * 0.20),
      Activities: Math.round(totalBudget * 0.15),
      Shopping: Math.round(totalBudget * 0.15)
    });
  };

  const budget = budgetStatus?.categories || {};
  const totalSpent = budgetStatus?.totalSpent || 0;
  const overallPct = budgetStatus ? Math.round((totalSpent / (budgetStatus.totalBudget || 1)) * 100) : 0;

  // AI Suggestions
  const suggestions = useMemo(() => {
    if (!budgetStatus) return [];
    const tips = [];
    const cats = budgetStatus.categories || {};

    Object.entries(cats).forEach(([key, val]) => {
      if (val.progress > 85) {
        tips.push({ type: "danger", text: `${key} budget critically low — ${val.progress}% used!` });
      } else if (val.progress > 60) {
        tips.push({ type: "warning", text: `${key} spending at ${val.progress}% — monitor closely.` });
      }
    });

    // Cross-category suggestions
    const stayData = cats.Stay || cats.Hotels;
    const shopData = cats.Shopping;
    if (stayData && shopData && shopData.remaining > 2000 && stayData.progress > 75) {
      tips.push({ type: "suggestion", text: `Shift ₹2,000 from Shopping to Stay for a better hotel upgrade.` });
    }

    if (tips.length === 0) {
      tips.push({ type: "success", text: "Budget allocation looks healthy. All categories within safe limits." });
    }
    return tips;
  }, [budgetStatus]);

  return (
    <FeatureShell
      feature="Finance"
      title="Smart Budget Manager"
      subtitle="Category tracking with AI insights"
      icon="◈"
      loading={loading}
      error={error}
      action={
        <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
          <button
            className="button button-primary"
            type="button"
            onClick={() => onCreateBudget({ totalBudget, allocations })}
            disabled={loading}
          >
            {loading ? "Saving..." : budgetStatus ? "Update Budget" : "Create Budget"}
          </button>
          <button className="button button-ghost" type="button" onClick={resetToSuggested}>
            AI Suggest
          </button>
        </div>
      }
    >
      <div className="budget-dashboard-v2">
        {/* Total Budget Input */}
        <div className="budget-total-row">
          <label className="budget-total-input">
            <span>Total Trip Budget</span>
            <input
              type="number"
              min="1000"
              step="500"
              value={totalBudget}
              onChange={e => setTotalBudget(Number(e.target.value) || 0)}
            />
          </label>
          <div className="budget-total-remaining" style={{ color: remaining >= 0 ? "var(--green)" : "var(--red)" }}>
            <span>Unallocated</span>
            <strong>{formatMoney(Math.abs(remaining))}</strong>
            {remaining < 0 && <small className="budget-over-label">Over-allocated!</small>}
          </div>
        </div>

        {/* Category Sliders */}
        <div className="budget-sliders-section">
          <div className="budget-sliders-header">
            <h4>Budget Allocation</h4>
            <button
              className={`budget-chart-toggle ${showChart ? "active" : ""}`}
              onClick={() => setShowChart(p => !p)}
            >
              {showChart ? "Hide Chart" : "Show Chart"}
            </button>
          </div>

          {/* Donut Chart */}
          <AnimatePresence>
            {showChart && (
              <motion.div
                className="budget-donut-wrapper"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.3 }}
              >
                <div className="budget-donut">
                  <svg viewBox="0 0 120 120" className="budget-donut-svg">
                    {(() => {
                      let offset = 0;
                      const total = allocatedTotal || 1;
                      return categories.map(cat => {
                        const pct = (allocations[cat.id] || 0) / total;
                        const dashArray = pct * 314.16; // 2 * PI * 50
                        const dashOffset = -offset * 314.16;
                        offset += pct;
                        return (
                          <circle
                            key={cat.id}
                            cx="60" cy="60" r="50"
                            fill="none"
                            stroke={cat.color}
                            strokeWidth="12"
                            strokeDasharray={`${dashArray} ${314.16 - dashArray}`}
                            strokeDashoffset={dashOffset}
                            style={{ transition: "all 0.5s ease" }}
                          />
                        );
                      });
                    })()}
                  </svg>
                  <div className="budget-donut-center">
                    <strong>{formatMoney(allocatedTotal)}</strong>
                    <small>Allocated</small>
                  </div>
                </div>
                <div className="budget-donut-legend">
                  {categories.map(cat => (
                    <div key={cat.id} className="budget-legend-item">
                      <span className="budget-legend-dot" style={{ background: cat.color }} />
                      <span>{cat.id}</span>
                      <strong>{Math.round(((allocations[cat.id] || 0) / (allocatedTotal || 1)) * 100)}%</strong>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Slider Cards */}
          <div className="budget-slider-cards">
            {categories.map(cat => {
              const val = allocations[cat.id] || 0;
              const pct = Math.round((val / (totalBudget || 1)) * 100);
              return (
                <motion.div
                  key={cat.id}
                  className="budget-slider-card"
                  whileHover={{ y: -2 }}
                >
                  <div className="budget-slider-card-head">
                    <span className="budget-slider-card-icon">{cat.icon}</span>
                    <span className="budget-slider-card-name">{cat.id}</span>
                    <span className="budget-slider-card-amount">{formatMoney(val)}</span>
                    <span className="budget-slider-card-pct" style={{ color: cat.color }}>{pct}%</span>
                  </div>
                  <div className="budget-slider-track-wrapper">
                    <input
                      type="range"
                      min="0"
                      max={totalBudget}
                      step="100"
                      value={val}
                      onChange={e => handleSliderChange(cat.id, e.target.value)}
                      className="budget-range-input"
                      style={{ "--slider-color": cat.color, "--slider-pct": `${pct}%` }}
                    />
                  </div>
                  <div className="budget-slider-bar">
                    <motion.div
                      className="budget-slider-bar-fill"
                      style={{ background: cat.color }}
                      animate={{ width: `${Math.min(pct, 100)}%` }}
                      transition={{ duration: 0.4, ease: "easeOut" }}
                    />
                  </div>
                </motion.div>
              );
            })}
          </div>
        </div>

        {/* Live Budget Status */}
        {budgetStatus && (
          <motion.div
            className="budget-live-section"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35 }}
          >
            {/* Overall Health Bar */}
            <div className="budget-health-bar">
              <div className="budget-health-header">
                <span>Overall Spend</span>
                <span style={{ color: getHealthColor(overallPct), fontWeight: 700 }}>
                  {overallPct}% — {getHealthLabel(overallPct)}
                </span>
              </div>
              <div className="budget-health-track">
                <motion.div
                  className="budget-health-fill"
                  style={{ background: getHealthColor(overallPct) }}
                  animate={{ width: `${Math.min(overallPct, 100)}%` }}
                  transition={{ duration: 0.5 }}
                />
              </div>
              <div className="budget-health-meta">
                <span>Spent: {formatMoney(totalSpent)}</span>
                <span>Remaining: {formatMoney((budgetStatus.totalBudget || 0) - totalSpent)}</span>
              </div>
            </div>

            {/* Category Status Cards */}
            <div className="budget-category-grid">
              {categories.map(cat => {
                const catKey = cat.id === "Stay" ? (budget.Stay ? "Stay" : "Hotels") : cat.id;
                const item = budget[catKey] || { allocated: 0, spent: 0, remaining: 0, progress: 0 };
                const healthColor = getHealthColor(item.progress);
                return (
                  <motion.article
                    key={cat.id}
                    className="budget-cat-card"
                    whileHover={{ y: -3, scale: 1.01 }}
                  >
                    <div className="budget-cat-header">
                      <span className="budget-cat-icon">{cat.icon}</span>
                      <span className="budget-cat-name">{cat.id}</span>
                      <span className="budget-cat-badge" style={{ color: healthColor, borderColor: healthColor + "40" }}>
                        {item.progress}%
                      </span>
                    </div>
                    <div className="budget-cat-progress">
                      <motion.div
                        className="budget-cat-progress-fill"
                        style={{ background: healthColor }}
                        animate={{ width: `${Math.min(item.progress, 100)}%` }}
                        transition={{ duration: 0.5 }}
                      />
                    </div>
                    <div className="budget-cat-metrics">
                      <div>
                        <span>Allocated</span>
                        <strong>{formatMoney(item.allocated)}</strong>
                      </div>
                      <div>
                        <span>Spent</span>
                        <strong>{formatMoney(item.spent)}</strong>
                      </div>
                      <div>
                        <span>Left</span>
                        <strong style={{ color: item.remaining < 0 ? "var(--red)" : "inherit" }}>
                          {formatMoney(item.remaining)}
                        </strong>
                      </div>
                    </div>
                    {item.progress > 85 && (
                      <div className="budget-cat-warning">
                        ⚠️ {cat.id} budget low!
                      </div>
                    )}
                  </motion.article>
                );
              })}
            </div>

            {/* Expense Form */}
            <div className="budget-expense-section">
              <h4>Add Expense</h4>
              <div className="budget-expense-row">
                <select value={expenseCategory} onChange={e => setExpenseCategory(e.target.value)}>
                  {categories.map(cat => (
                    <option key={cat.id} value={cat.id}>{cat.icon} {cat.id}</option>
                  ))}
                </select>
                <input
                  type="number"
                  min="0"
                  step="50"
                  value={expenseAmount}
                  onChange={e => setExpenseAmount(Number(e.target.value) || 0)}
                  placeholder="Amount"
                />
                <button
                  className="button button-success"
                  onClick={() => onAddExpense(expenseCategory === "Stay" ? "Hotels" : expenseCategory, expenseAmount)}
                  disabled={loading}
                >
                  + Add
                </button>
              </div>
            </div>
          </motion.div>
        )}

        {/* AI Suggestions */}
        {suggestions.length > 0 && budgetStatus && (
          <motion.div
            className="budget-suggestions"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
          >
            <h4>💡 AI Budget Insights</h4>
            {suggestions.map((tip, i) => (
              <motion.div
                key={i}
                className={`budget-suggestion-card ${tip.type}`}
                initial={{ opacity: 0, x: -12 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.1 * i }}
              >
                <span className="budget-suggestion-icon">
                  {tip.type === "danger" ? "🔴" : tip.type === "warning" ? "⚠️" : tip.type === "suggestion" ? "💡" : "✅"}
                </span>
                <span>{tip.text}</span>
              </motion.div>
            ))}
          </motion.div>
        )}

        {/* Empty State */}
        {!budgetStatus && (
          <div className="empty-feature-state">
            <strong>No budget plan created yet</strong>
            <span>Set your total budget, adjust category sliders, then click "Create Budget" to start tracking.</span>
          </div>
        )}
      </div>
    </FeatureShell>
  );
}

export default BudgetDashboard;
