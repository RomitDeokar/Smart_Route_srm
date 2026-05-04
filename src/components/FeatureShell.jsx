import { AnimatePresence, motion } from "framer-motion";
import { useState } from "react";

function FeatureShell({
  feature,
  title,
  subtitle,
  icon,
  loading,
  error,
  action,
  defaultExpanded = true,
  children
}) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  return (
    <motion.section
      layout
      className={`feature-shell glass-panel ${isExpanded ? "expanded" : "collapsed"}`}
      initial={{ opacity: 0, y: 18 }}
      whileInView={{ opacity: 1, y: 0 }}
      whileHover={{ y: -4, scale: 1.01 }}
      viewport={{ once: true, amount: 0.2 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
    >
      <div className="feature-shell-glow" />

      <motion.button
        type="button"
        className="feature-shell-header"
        onClick={() => setIsExpanded(current => !current)}
        whileTap={{ scale: 0.99 }}
      >
        <div className="feature-shell-icon">{icon}</div>
        <div className="feature-shell-copy">
          <p className="eyebrow">{feature}</p>
          <h3>{title}</h3>
          <span>{subtitle}</span>
        </div>
        <div className="feature-shell-side">
          {loading ? <span className="feature-shell-loading">Running</span> : null}
          <span className={`feature-shell-toggle ${isExpanded ? "open" : ""}`}>⌄</span>
        </div>
      </motion.button>

      {action ? <div className="feature-shell-action">{action}</div> : null}
      {error ? <div className="feature-error">{error}</div> : null}

      <AnimatePresence initial={false}>
        {isExpanded ? (
          <motion.div
            key="content"
            layout
            className="feature-shell-content"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.28, ease: "easeOut" }}
          >
            {loading ? (
              <div className="skeleton-stack">
                <span className="skeleton-line wide" />
                <span className="skeleton-line" />
                <span className="skeleton-line short" />
              </div>
            ) : null}
            {children}
          </motion.div>
        ) : null}
      </AnimatePresence>
    </motion.section>
  );
}

export default FeatureShell;
