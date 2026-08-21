import { useCallback, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, X, FileText, Cpu, CheckCircle, AlertCircle, Zap } from "lucide-react";

export default function UploadZone({ files, onAddFiles, onRemoveFile, onUpload, status, progress, stats, error }) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    const dropped = Array.from(e.dataTransfer.files).filter((f) => f.type === "application/pdf");
    if (dropped.length) onAddFiles(dropped);
  }, [onAddFiles]);

  const handleFileInput = (e) => {
    const selected = Array.from(e.target.files || []);
    if (selected.length) onAddFiles(selected);
    e.target.value = "";
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-6 relative z-10">
      {/* Hero */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-10"
      >
        <div className="flex items-center justify-center gap-3 mb-4">
          <div className="w-12 h-12 rounded-2xl glass border border-cyan-400/30 flex items-center justify-center">
            <Zap size={22} className="text-cyan-400" />
          </div>
          <div className="text-left">
            <h1 className="text-2xl font-bold text-white glow-text tracking-tight">
              PDF Intelligence
            </h1>
            <p className="text-xs text-cyan-400/70 font-mono tracking-widest uppercase">
              Agentic RAG Platform
            </p>
          </div>
        </div>
        <p className="text-slate-400 text-sm max-w-md">
          Upload one or more PDFs. Our agentic system will parse, chunk, embed, and index them — then you can ask anything.
        </p>
      </motion.div>

      {/* Drop zone */}
      <motion.div
        initial={{ opacity: 0, scale: 0.97 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.1 }}
        className="w-full max-w-lg"
      >
        <label
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          className={`glass glass-hover rounded-2xl border-2 border-dashed flex flex-col items-center justify-center p-10 cursor-pointer transition-all duration-300
            ${isDragging ? "drop-zone-active" : "border-slate-700 hover:border-cyan-500/40"}`}
        >
          <input
            type="file"
            multiple
            accept=".pdf"
            className="hidden"
            onChange={handleFileInput}
          />
          <motion.div
            animate={{ y: isDragging ? -6 : 0 }}
            transition={{ type: "spring", stiffness: 400 }}
          >
            <Upload size={36} className={`mb-4 transition-colors ${isDragging ? "text-cyan-400" : "text-slate-500"}`} />
          </motion.div>
          <p className="text-slate-300 font-medium text-sm">
            {isDragging ? "Release to drop" : "Drop PDFs here"}
          </p>
          <p className="text-slate-500 text-xs mt-1">or click to browse</p>
        </label>

        {/* File pills */}
        <AnimatePresence>
          {files.length > 0 && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-4 space-y-2"
            >
              {files.map((f) => (
                <motion.div
                  key={f.name}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 10 }}
                  className="glass rounded-xl px-3 py-2 flex items-center gap-2"
                >
                  <FileText size={14} className="text-cyan-400 shrink-0" />
                  <span className="text-sm text-slate-300 truncate flex-1">{f.name}</span>
                  <span className="text-[10px] text-slate-500 font-mono shrink-0">
                    {(f.size / 1024).toFixed(0)} KB
                  </span>
                  <button
                    onClick={() => onRemoveFile(f.name)}
                    className="text-slate-500 hover:text-red-400 transition-colors ml-1"
                  >
                    <X size={12} />
                  </button>
                </motion.div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Progress bar */}
        {status === "uploading" && (
          <div className="mt-4">
            <div className="flex justify-between text-xs text-slate-400 mb-1 font-mono">
              <span className="flex items-center gap-1.5">
                <Cpu size={11} className="animate-spin" /> Processing documents...
              </span>
              <span>{Math.round(progress)}%</span>
            </div>
            <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
              <motion.div
                className="h-full rounded-full"
                style={{ background: "linear-gradient(90deg, #00ffcc, #bf5fff)" }}
                animate={{ width: `${progress}%` }}
                transition={{ duration: 0.4 }}
              />
            </div>
          </div>
        )}

        {/* Success stats */}
        <AnimatePresence>
          {status === "done" && stats && (
            <motion.div
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-4 glass rounded-xl p-3 border border-cyan-400/20"
            >
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle size={14} className="text-cyan-400" />
                <span className="text-sm text-cyan-300 font-medium">Indexed successfully</span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-[11px] font-mono text-slate-400">
                <span>Parent chunks: <b className="text-slate-200">{stats.parent_chunks}</b></span>
                <span>Child chunks: <b className="text-slate-200">{stats.child_chunks}</b></span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Error */}
        {status === "error" && error && (
          <div className="mt-4 glass rounded-xl p-3 border border-red-500/30 flex items-start gap-2">
            <AlertCircle size={14} className="text-red-400 shrink-0 mt-0.5" />
            <p className="text-xs text-red-300">{error}</p>
          </div>
        )}

        {/* Upload button */}
        {files.length > 0 && status !== "uploading" && (
          <motion.button
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            whileTap={{ scale: 0.97 }}
            onClick={onUpload}
            className="btn-neon w-full mt-4 py-3 rounded-xl text-sm font-semibold tracking-wide"
          >
            {status === "done" ? "Re-index Documents" : `Index ${files.length} PDF${files.length > 1 ? "s" : ""}`}
          </motion.button>
        )}
      </motion.div>
    </div>
  );
}
