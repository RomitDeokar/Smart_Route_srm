import { useState } from "react";
import { motion } from "framer-motion";

export default function PayButton({ items, metadata = {}, label = "Book Now", style = {}, onSuccess, addToast }) {
  const [loading, setLoading] = useState(false);

  const handleClick = async () => {
    if (loading || !items?.length) return;
    setLoading(true);
    try {
      const res  = await fetch("/api/payments/checkout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ items, metadata }),
      });
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || "Checkout failed");

      if (data.mock) {
        addToast?.("Demo: Payment flow started (mock mode)", "info");
        setTimeout(() => {
          onSuccess?.({ sessionId: data.sessionId, mock: true });
          addToast?.("Demo booking confirmed!", "success");
        }, 1200);
      } else {
        window.location.href = data.url;
      }
    } catch (e) {
      addToast?.(e.message || "Payment error", "error");
    } finally {
      setLoading(false);
    }
  };

  const total = items?.reduce((s, i) => s + (i.price || 0) * (i.quantity || 1), 0) || 0;

  return (
    <motion.button
      className="btn btn-primary"
      style={style}
      onClick={handleClick}
      disabled={loading}
      whileHover={{ scale: loading ? 1 : 1.02 }}
      whileTap={{ scale: 0.98 }}
    >
      {loading ? "Processing..." : `${label}${total > 0 ? ` — ₹${total.toLocaleString("en-IN")}` : ""}`}
    </motion.button>
  );
}
