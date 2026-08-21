import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown, FileText, Hash, Layers } from "lucide-react";

function ScoreBar({ label, value, color }) {
  const pct =
    value === null || value === undefined
      ? 0
      : label === "BM25"
      ? Math.min((value / 15) * 100, 100)
      : label === "Rerank"
      ? Math.max(0, ((value + 15) / 15) * 100)
      : Math.min(value * 100, 100);

  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] font-mono text-slate-400 w-12 shrink-0">{label}</span>
      <div className="score-bar-track flex-1">
        <div className="score-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="text-[10px] font-mono text-slate-300 w-12 text-right shrink-0">
        {value !== null && value !== undefined ? Number(value).toFixed(3) : "—"}
      </span>
    </div>
  );
}

function FragmentCard({ doc, index }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="glass rounded-lg overflow-hidden text-sm border border-white/5">
      {/* Card header */}
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between px-3 py-2 text-left hover:bg-white/3 transition-colors"
      >
        <div className="flex items-center gap-2 min-w-0">
          <FileText size={12} className="text-cyan-400 shrink-0" />
          <span className="text-slate-200 text-xs font-medium truncate">
            Fragment {index + 1}
          </span>
          {doc.document_name && (
            <span className="text-[10px] bg-cyan-400/10 text-cyan-400 px-1.5 py-0.5 rounded font-mono truncate max-w-[100px]">
              {doc.document_name}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 shrink-0 ml-2">
          {doc.page && (
            <span className="text-[10px] text-slate-500 font-mono">p.{doc.page}</span>
          )}
          <ChevronDown
            size={12}
            className={`text-slate-400 transition-transform duration-200 ${open ? "rotate-180" : ""}`}
          />
        </div>
      </button>

      {/* Score bars (always visible) */}
      <div className="px-3 pb-2 space-y-1">
        <ScoreBar label="Dense"  value={doc.dense_score}  color="linear-gradient(90deg,#00ffcc,#00e5b7)" />
        <ScoreBar label="BM25"   value={doc.bm25_score}   color="linear-gradient(90deg,#bf5fff,#9b33ff)" />
        <ScoreBar label="RRF"    value={doc.rrf_score}    color="linear-gradient(90deg,#60a5fa,#3b82f6)" />
        <ScoreBar label="Rerank" value={doc.rerank_score} color="linear-gradient(90deg,#fb923c,#f97316)" />
      </div>

      {/* Expandable content */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-3 pb-3 pt-2 border-t border-white/5">
              {doc.section && (
                <p className="text-[10px] text-purple-400 font-mono mb-1.5 flex items-center gap-1">
                  <Hash size={10} />§{doc.section}
                </p>
              )}
              <p className="text-slate-300 text-xs leading-relaxed whitespace-pre-wrap">
                {doc.content}
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function SourceFragments({ sources }) {
  const [open, setOpen] = useState(false);
  if (!sources?.length) return null;

  return (
    <div className="mt-3">
      {/* Single top-level toggle button */}
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center gap-2 px-3 py-2 glass glass-hover rounded-xl text-left transition-all"
      >
        <Layers size={13} className="text-cyan-400 shrink-0" />
        <span className="text-xs font-semibold text-cyan-300 flex-1">
          Fragments
        </span>
        <span className="text-[10px] font-mono text-slate-500 mr-1">
          {sources.length} retrieved
        </span>
        <ChevronDown
          size={13}
          className={`text-slate-400 transition-transform duration-200 ${open ? "rotate-180" : ""}`}
        />
      </button>

      {/* Staggered cards */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <div className="mt-2 space-y-2">
              {sources.map((doc, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -8 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.06 }}
                >
                  <FragmentCard doc={doc} index={i} />
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
