import { AnimatePresence, motion } from "framer-motion";

const destinations = [
  { name: "Shillong", emoji: "🏔️", region: "Northeast India", desc: "Misty highlands and living root bridges.", vibe: "Nature" },
  { name: "Goa", emoji: "🏖️", region: "Western Coast", desc: "Sun-kissed beaches and Portuguese heritage.", vibe: "Beach" },
  { name: "Ooty", emoji: "🌿", region: "Tamil Nadu", desc: "Rolling tea estates and mountain air.", vibe: "Hills" },
  { name: "Munnar", emoji: "🍃", region: "Kerala", desc: "Emerald tea gardens and misty mornings.", vibe: "Retreat" },
  { name: "Coorg", emoji: "☕", region: "Karnataka", desc: "Coffee plantations and waterfalls.", vibe: "Culture" },
  { name: "Wayanad", emoji: "🌳", region: "Kerala", desc: "Dense forests and spice-scented trails.", vibe: "Adventure" },
  { name: "Rishikesh", emoji: "🧘", region: "Uttarakhand", desc: "Sacred rivers and adventure sports.", vibe: "Spiritual" },
  { name: "Udaipur", emoji: "🏰", region: "Rajasthan", desc: "Lake palaces and royal cuisine.", vibe: "Heritage" }
];

function DestinationModal({ isOpen, onClose, onSelect }) {
  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          className="dest-modal-overlay"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.25 }}
          onClick={onClose}
        >
          <motion.div
            className="dest-modal"
            initial={{ opacity: 0, y: 40, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 30, scale: 0.96 }}
            transition={{ duration: 0.35, ease: "easeOut" }}
            onClick={e => e.stopPropagation()}
          >
            <h2>Choose a destination</h2>

            <div className="dest-grid">
              {destinations.map((dest, index) => (
                <motion.button
                  key={dest.name}
                  className="dest-card"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.06 * index, duration: 0.3 }}
                  whileHover={{ y: -4 }}
                  whileTap={{ scale: 0.97 }}
                  onClick={() => {
                    onSelect(dest.name);
                    onClose();
                  }}
                >
                  <div className="dest-card-icon">{dest.emoji}</div>
                  <div className="dest-card-name">{dest.name}</div>
                  <div className="dest-card-vibe">{dest.vibe}</div>
                </motion.button>
              ))}
            </div>

            <button className="button button-ghost" onClick={onClose} style={{ width: "100%" }}>
              Close
            </button>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

export default DestinationModal;
