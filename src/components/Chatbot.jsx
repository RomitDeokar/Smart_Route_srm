import { useMemo, useState } from "react";

const starterPrompts = [
  "Plan a 3 day trip to Goa",
  "Suggest nearby places I can reach in 2 hours",
  "Find cheaper hotels"
];

function buildContextSummary(context) {
  return [
    context.origin && `From ${context.origin}`,
    context.destination && `to ${context.destination}`,
    context.days && `${context.days} day(s)`,
    context.budget && `budget ${context.budget}`,
    context.persona && `${context.persona} persona`
  ]
    .filter(Boolean)
    .join(" • ");
}

function Chatbot({ context }) {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    {
      id: "assistant-welcome",
      role: "assistant",
      content: "I’m your AI travel assistant. Ask for itineraries, nearby quick trips, or cheaper hotel ideas.",
      cards: [
        {
          type: "capabilities",
          title: "I can help with",
          items: ["Trip planning", "Quick nearby suggestions", "Budget-aware hotel guidance"]
        }
      ],
      quickActions: starterPrompts,
      intent: "welcome"
    }
  ]);
  const [draft, setDraft] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState("");

  const historyPayload = useMemo(
    () =>
      messages.slice(-6).map(message => ({
        role: message.role,
        content: message.content
      })),
    [messages]
  );

  const sendMessage = async rawMessage => {
    const trimmed = rawMessage.trim();
    if (!trimmed || isSending) return;

    const userMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: trimmed
    };

    setMessages(current => [...current, userMessage]);
    setDraft("");
    setError("");
    setIsSending(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: trimmed,
          context,
          history: [...historyPayload, { role: "user", content: trimmed }]
        })
      });

      const data = await response.json();
      if (!response.ok || !data.ok) {
        throw new Error(data.error || "Assistant request failed.");
      }

      setMessages(current => [
        ...current,
        {
          id: `assistant-${Date.now()}`,
          role: "assistant",
          content: data.reply,
          cards: data.cards || [],
          quickActions: data.quickActions || [],
          intent: data.intent || "general"
        }
      ]);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsSending(false);
    }
  };

  const handleSubmit = event => {
    event.preventDefault();
    sendMessage(draft);
  };

  const contextSummary = buildContextSummary(context);

  return (
    <>
      <button
        type="button"
        className="chatbot-launcher"
        aria-label={isOpen ? "Close AI travel assistant" : "Open AI travel assistant"}
        onClick={() => setIsOpen(current => !current)}
      >
        {isOpen ? "×" : "AI"}
      </button>

      <aside className={`chatbot-panel glass-panel ${isOpen ? "open" : ""}`} aria-hidden={!isOpen}>
        <div className="chatbot-header">
          <div>
            <p className="eyebrow">AI Assistant</p>
            <h3>Travel Agent Console</h3>
          </div>
          <button type="button" className="chatbot-close" onClick={() => setIsOpen(false)} aria-label="Close assistant">
            ×
          </button>
        </div>

        <div className="chatbot-context">
          <strong>Current context</strong>
          <span>{contextSummary || "No active trip context yet."}</span>
        </div>

        <div className="chatbot-prompts">
          {starterPrompts.map(prompt => (
            <button key={prompt} type="button" className="chatbot-prompt" onClick={() => sendMessage(prompt)}>
              {prompt}
            </button>
          ))}
        </div>

        <div className="chatbot-messages">
          {messages.map(message => (
            <article key={message.id} className={`chat-message ${message.role}`}>
              <div className="chat-message-bubble">
                <small>{message.role === "assistant" ? "AI Travel Agent" : "You"}</small>
                <p>{message.content}</p>

                {message.cards?.length ? (
                  <div className="chat-cards">
                    {message.cards.map(card => (
                      <section key={`${message.id}-${card.title}`} className="chat-card">
                        <strong>{card.title}</strong>
                        <div className="chat-card-items">
                          {card.items?.map(item => (
                            <span key={`${card.title}-${item}`}>{item}</span>
                          ))}
                        </div>
                      </section>
                    ))}
                  </div>
                ) : null}

                {message.quickActions?.length ? (
                  <div className="chat-quick-actions">
                    {message.quickActions.map(action => (
                      <button key={`${message.id}-${action}`} type="button" onClick={() => sendMessage(action)}>
                        {action}
                      </button>
                    ))}
                  </div>
                ) : null}
              </div>
            </article>
          ))}

          {isSending ? (
            <article className="chat-message assistant">
              <div className="chat-message-bubble typing">
                <small>AI Travel Agent</small>
                <p>Thinking through your trip...</p>
              </div>
            </article>
          ) : null}
        </div>

        {error ? <div className="chatbot-error">{error}</div> : null}

        <form className="chatbot-form" onSubmit={handleSubmit}>
          <textarea
            rows="2"
            placeholder="Ask for itineraries, hotels, nearby places, or budget help..."
            value={draft}
            onChange={event => setDraft(event.target.value)}
            onKeyDown={event => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                sendMessage(draft);
              }
            }}
          />
          <button type="submit" className="button button-primary" disabled={isSending || !draft.trim()}>
            Send
          </button>
        </form>
      </aside>
    </>
  );
}

export default Chatbot;
