import { motion } from "framer-motion";
import { FileText, Upload, Trash2, Zap, ChevronDown } from "lucide-react";
import { useState } from "react";

export default function Sidebar({ files, models, selectedModel, onModelChange, onNewUpload, onClearChat }) {
  const [modelOpen, setModelOpen] = useState(false);

  const shortName = (m) => m.split("/").pop().replace(/-/g, " ");

  return (
    <div className="flex flex-col h-full p-4 gap-4">
      {/* Logo */}
      <div className="flex items-center gap-2.5 px-1 py-2">
        <div className="w-8 h-8 rounded-xl glass border border-cyan-400/30 flex items-center justify-center">
          <Zap size={16} className="text-cyan-400" />
        </div>
        <div>
          <p className="text-sm font-bold text-white leading-tight">PDF Intelligence</p>
          <p className="text-[10px] font-mono text-cyan-400/60 tracking-widest">AGENTIC RAG</p>
        </div>
      </div>

      <div className="h-px bg-white/5" />

      {/* Model selector */}
      <div>
        <p className="text-[10px] font-mono text-slate-500 uppercase tracking-widest mb-2">Model</p>
        <div className="relative">
          <button
            onClick={() => setModelOpen((o) => !o)}
            className="glass glass-hover w-full rounded-xl px-3 py-2.5 flex items-center justify-between text-sm text-slate-200"
          >
            <span className="truncate font-mono text-xs text-cyan-400">{shortName(selectedModel)}</span>
            <ChevronDown size={14} className={`text-slate-400 shrink-0 ml-2 transition-transform ${modelOpen ? "rotate-180" : ""}`} />
          </button>

          {modelOpen && (
            <motion.div
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              className="absolute top-full left-0 right-0 z-50 mt-1 glass rounded-xl border border-white/10 overflow-hidden shadow-2xl"
            >
              {models.map((m) => (
                <button
                  key={m}
                  onClick={() => { onModelChange(m); setModelOpen(false); }}
                  className={`w-full text-left px-3 py-2 text-xs font-mono transition-colors
                    ${m === selectedModel
                      ? "text-cyan-400 bg-cyan-400/10"
                      : "text-slate-300 hover:bg-white/5 hover:text-white"
                    }`}
                >
                  {m}
                </button>
              ))}
            </motion.div>
          )}
        </div>
      </div>

      {/* Indexed files */}
      {files.length > 0 && (
        <div>
          <p className="text-[10px] font-mono text-slate-500 uppercase tracking-widest mb-2">
            Indexed ({files.length})
          </p>
          <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
            {files.map((f) => (
              <div key={f.name} className="glass rounded-lg px-2.5 py-1.5 flex items-center gap-2">
                <FileText size={11} className="text-cyan-400 shrink-0" />
                <span className="text-[11px] text-slate-300 truncate">{f.name}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="flex-1" />

      {/* Actions */}
      <div className="space-y-2">
        <button
          onClick={onClearChat}
          className="w-full glass glass-hover rounded-xl px-3 py-2 text-xs text-slate-400 hover:text-slate-200 flex items-center gap-2 transition-colors"
        >
          <Trash2 size={12} /> Clear chat
        </button>
        <button
          onClick={onNewUpload}
          className="btn-neon w-full rounded-xl px-3 py-2 text-xs flex items-center gap-2 justify-center"
        >
          <Upload size={12} /> New upload
        </button>
      </div>
    </div>
  );
}
