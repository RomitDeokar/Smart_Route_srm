import { AnimatePresence, motion } from "framer-motion";
const ICONS = { success:"✓", error:"✕", info:"i", warning:"⚠" };
const COLORS = { success:"var(--green)", error:"var(--red)", info:"var(--blue)", warning:"var(--amber)" };
export default function Toast({ toasts }) {
  return (
    <div className="toast-stack">
      <AnimatePresence>
        {toasts.map(t=>(
          <motion.div key={t.id} className={`toast-item ${t.type}`} initial={{opacity:0,x:20,scale:0.96}} animate={{opacity:1,x:0,scale:1}} exit={{opacity:0,x:20,scale:0.95}} transition={{duration:0.22,ease:"easeOut"}}>
            <span style={{fontSize:12,fontWeight:700,flexShrink:0,color:COLORS[t.type]||"var(--blue)"}}>{ICONS[t.type]||"i"}</span>
            <span style={{color:"var(--text)",fontSize:13}}>{t.msg}</span>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}
