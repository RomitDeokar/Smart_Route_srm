/* useAgentStream.js — WebSocket hook for live agent pipeline updates */
import { useEffect, useRef, useState, useCallback } from "react";

export default function useAgentStream() {
  const [connected,   setConnected]   = useState(false);
  const [agentEvents, setAgentEvents] = useState([]);
  const [lastEvent,   setLastEvent]   = useState(null);
  const wsRef = useRef(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host     = window.location.hostname;
    const port     = import.meta.env.DEV ? "8787" : window.location.port;
    const url      = `${protocol}//${host}:${port}`;

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen    = () => setConnected(true);
      ws.onclose   = () => { setConnected(false); wsRef.current = null; };
      ws.onerror   = () => { setConnected(false); };

      ws.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data);
          setLastEvent(data);

          if (data.type === "agent_update") {
            setAgentEvents(prev => [
              { ...data, id: Date.now() + Math.random() },
              ...prev
            ].slice(0, 20));
          }
        } catch { /* ignore malformed */ }
      };
    } catch {
      /* WebSocket not available — silent fail */
    }
  }, []);

  const disconnect = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
  }, []);

  const clearEvents = useCallback(() => setAgentEvents([]), []);

  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  return { connected, agentEvents, lastEvent, clearEvents };
}
