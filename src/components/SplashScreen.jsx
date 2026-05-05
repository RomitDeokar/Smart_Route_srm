import { motion } from "framer-motion";
export default function SplashScreen() {
  return (
    <div className="splash-screen">
      <div className="splash-inner">
        <motion.div className="splash-logo" initial={{scale:0.5,opacity:0}} animate={{scale:1,opacity:1}} transition={{duration:0.5,ease:[0.34,1.56,0.64,1]}}>SR</motion.div>
        <motion.div className="splash-name" initial={{opacity:0,y:10}} animate={{opacity:1,y:0}} transition={{delay:0.25}}>SmartRoute</motion.div>
        <motion.p initial={{opacity:0}} animate={{opacity:1}} transition={{delay:0.4}} style={{fontSize:11.5,color:"var(--text-3)",letterSpacing:"0.08em",textTransform:"uppercase"}}>SRMIST Chapter</motion.p>
        <motion.div className="splash-bar" initial={{opacity:0}} animate={{opacity:1}} transition={{delay:0.5}}><div className="splash-bar-fill"/></motion.div>
        <motion.p initial={{opacity:0}} animate={{opacity:1}} transition={{delay:0.6}} style={{fontSize:12.5,color:"var(--text-2)"}}>Initializing AI agents...</motion.p>
      </div>
    </div>
  );
}
