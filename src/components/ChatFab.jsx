import { useState, useRef, useEffect } from "react";
import { AnimatePresence, motion } from "framer-motion";

const STARTERS = [
  "Plan a 3 day trip to Goa",
  "Find cheap hotels in Shillong",
  "Show budget breakdown",
];

export default function ChatFab({ tripCtx, addToast }) {
  const [open, setOpen]         = useState(false);
  const [messages, setMessages] = useState([{
    id: "w0", role: "assistant",
    content: "Hi! Ask me anything about your trip — hotels, flights, itineraries, or packing.",
  }]);
  const [draft, setDraft]   = useState("");
  const [sending, setSending] = useState(false);
  const bottomRef             = useRef(null);

  useEffect(() => {
    if (open) bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, open]);

  const send = async (text) => {
    const t = (text || draft).trim();
    if (!t || sending) return;
    setMessages(m => [...m, { id: `u${Date.now()}`, role: "user", content: t }]);
    setDraft("");
    setSending(true);
    try {
      const res  = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: t,
          context: tripCtx,
          history: messages.slice(-6).map(m => ({ role: m.role, content: m.content })),
        }),
      });
      const data = await res.json();
      if (!data.ok) throw new Error(data.error);
      setMessages(m => [...m, {
        id: `a${Date.now()}`, role: "assistant",
        content: data.reply, quickActions: data.quickActions || [],
      }]);
    } catch (e) {
      addToast(e.message, "error");
    } finally {
      setSending(false);
    }
  };

  return (
    <>
      <AnimatePresence>
        {open && (
          <motion.div
            className="chat-popup"
            initial={{ opacity: 0, y: 16, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 12, scale: 0.97 }}
            transition={{ duration: 0.22 }}
          >
            {/* Header */}
            <div className="chat-popup-header">
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <div style={{ width: 28, height: 28, borderRadius: "var(--r-sm)", background: "var(--g-blue)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
                  </svg>
                </div>
                <span className="chat-popup-title">Travel Assistant</span>
                <span className="pill green" style={{ fontSize: 10 }}>
                  <span className="dot-live" />Live
                </span>
              </div>
              <button onClick={() => setOpen(false)} style={{ background: "none", cursor: "pointer", color: "var(--text-3)", fontSize: 20, lineHeight: 1 }}>
                ×
              </button>
            </div>

            {/* Context */}
            <div style={{ padding: "6px 14px", background: "var(--bg-soft)", borderBottom: "1px solid var(--border)", fontSize: 11.5, color: "var(--text-3)", display: "flex", gap: 5, flexWrap: "wrap" }}>
              <span>{tripCtx.origin}</span><span>→</span><span>{tripCtx.destination}</span>
              <span>·</span><span>{tripCtx.days}d</span><span>·</span>
              <span>₹{(tripCtx.budget || 0).toLocaleString("en-IN")}</span>
            </div>

            {/* Quick starters */}
            <div style={{ padding: "8px 12px", display: "flex", gap: 5, flexWrap: "wrap", borderBottom: "1px solid var(--border)" }}>
              {STARTERS.map(s => (
                <button key={s} className="chat-quick-btn" style={{ fontSize: 11 }} onClick={() => send(s)}>
                  {s}
                </button>
              ))}
            </div>

            {/* Messages */}
            <div className="chat-messages" style={{ flex: 1, maxHeight: 280 }}>
              {messages.map(msg => (
                <div key={msg.id} className={`chat-msg ${msg.role}`}>
                  {msg.role === "assistant" && (
                    <div className="chat-avatar" style={{ width: 26, height: 26, fontSize: 10 }}>AI</div>
                  )}
                  <div>
                    <div className="chat-bubble" style={{ fontSize: 12.5, padding: "9px 12px" }}>{msg.content}</div>
                    {msg.quickActions?.length > 0 && (
                      <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 5 }}>
                        {msg.quickActions.slice(0, 2).map(qa => (
                          <button key={qa} className="chat-quick-btn" style={{ fontSize: 11 }} onClick={() => send(qa)}>{qa}</button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {sending && (
                <div className="chat-msg assistant">
                  <div className="chat-avatar" style={{ width: 26, height: 26, fontSize: 10 }}>AI</div>
                  <div className="chat-bubble" style={{ padding: "10px 14px", color: "var(--text-3)", fontSize: 12.5 }}>Thinking...</div>
                </div>
              )}
              <div ref={bottomRef} />
            </div>

            {/* Input */}
            <div className="chat-input-bar" style={{ padding: "10px 12px" }}>
              <textarea
                className="chat-textarea"
                rows={1}
                style={{ fontSize: 13 }}
                placeholder="Ask anything..."
                value={draft}
                onChange={e => setDraft(e.target.value)}
                onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
              />
              <button
                className="btn btn-primary"
                style={{ width: 36, height: 36, padding: 0, flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center" }}
                onClick={() => send()}
                disabled={sending || !draft.trim()}
              >
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="22" y1="2" x2="11" y2="13"/>
                  <polygon points="22 2 15 22 11 13 2 9 22 2"/>
                </svg>
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* FAB */}
      <button className="chat-fab" onClick={() => setOpen(o => !o)} aria-label="Open AI assistant">
        <div className="chat-fab-ring" />
        {open ? (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round">
            <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
          </svg>
        ) : (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
          </svg>
        )}
      </button>
    </>
  );
}
