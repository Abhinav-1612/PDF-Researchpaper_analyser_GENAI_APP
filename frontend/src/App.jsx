import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import UploadZone from "./components/UploadZone";
import Sidebar from "./components/Sidebar";
import ChatWindow from "./components/ChatWindow";
import { useUpload } from "./hooks/useUpload";
import { useChat } from "./hooks/useChat";

export default function App() {
  const {
    files, addFiles, removeFile, clearAll,
    upload, status, progress, stats, error,
    models, defaultModel, loadModels,
  } = useUpload();

  const [selectedModel, setSelectedModel]       = useState("");
  const [view, setView]                         = useState("upload");
  const [headerModelOpen, setHeaderModelOpen]   = useState(false);

  const {
    messages, isLoading, statusText, streamingText, streamingThink,
    sendMessage, cancelStream, clearChat
  } = useChat(selectedModel);

  // Load models on mount
  useEffect(() => { loadModels(); }, [loadModels]);

  // Set default model once loaded
  useEffect(() => {
    if (defaultModel && !selectedModel) setSelectedModel(defaultModel);
  }, [defaultModel]);

  // Transition to chat view once indexing is done
  useEffect(() => {
    if (status === "done") {
      const t = setTimeout(() => setView("chat"), 800);
      return () => clearTimeout(t);
    }
  }, [status]);

  const handleNewUpload = () => {
    clearAll();
    setView("upload");
  };

  return (
    <div className="relative h-screen overflow-hidden bg-navy-950">
      {/* Background orbs */}
      <div className="bg-orb bg-orb-1" />
      <div className="bg-orb bg-orb-2" />
      <div className="bg-orb bg-orb-3" />

      <AnimatePresence mode="wait">
        {view === "upload" ? (
          /* ── Upload screen ─────────────────────────────────────────── */
          <motion.div
            key="upload"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0, scale: 0.97 }}
            transition={{ duration: 0.3 }}
            className="relative z-10 h-full overflow-y-auto"
          >
            <UploadZone
              files={files}
              onAddFiles={addFiles}
              onRemoveFile={removeFile}
              onUpload={upload}
              status={status}
              progress={progress}
              stats={stats}
              error={error}
            />
          </motion.div>
        ) : (
          /* ── Chat screen ───────────────────────────────────────────── */
          <motion.div
            key="chat"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.35 }}
            className="relative z-10 h-full flex"
          >
            {/* Sidebar */}
            <aside className="hidden md:flex w-64 shrink-0 border-r border-white/5 flex-col">
              <Sidebar
                files={files}
                models={models}
                selectedModel={selectedModel}
                onModelChange={setSelectedModel}
                onNewUpload={handleNewUpload}
                onClearChat={clearChat}
              />
            </aside>

            {/* Chat panel */}
            <main className="flex-1 flex flex-col min-w-0 h-full">
              {/* Top bar */}
              <header className="shrink-0 flex items-center justify-between px-4 py-3 border-b border-white/5 gap-3">
                <div className="min-w-0">
                  <p className="text-sm font-semibold text-white truncate">Document Chat</p>
                  <p className="text-[11px] text-slate-500 font-mono">
                    {files.length} PDF{files.length !== 1 ? "s" : ""} indexed
                  </p>
                </div>

                {/* Always-visible model selector */}
                <div className="relative shrink-0" id="header-model-picker">
                  <button
                    onClick={() => setHeaderModelOpen((o) => !o)}
                    className="flex items-center gap-1.5 glass glass-hover rounded-xl px-3 py-1.5 text-xs font-mono text-cyan-400 border border-cyan-400/20 hover:border-cyan-400/40 transition-all"
                  >
                    <span className="max-w-[140px] truncate">{selectedModel?.split("/").pop()?.replace(/:free$/, "") || "Select Model"}</span>
                    <svg width="10" height="10" viewBox="0 0 10 10" className={`transition-transform ${headerModelOpen ? "rotate-180" : ""}`} fill="currentColor">
                      <path d="M1 3l4 4 4-4" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round"/>
                    </svg>
                  </button>

                  {headerModelOpen && (
                    <div className="absolute right-0 top-full mt-1 z-50 glass rounded-xl border border-white/10 shadow-2xl overflow-hidden min-w-[220px]">
                      {models.length === 0 && (
                        <p className="text-xs text-slate-500 px-3 py-2 font-mono">Loading models...</p>
                      )}
                      {models.map((m) => (
                        <button
                          key={m}
                          onClick={() => { setSelectedModel(m); setHeaderModelOpen(false); }}
                          className={`w-full text-left px-3 py-2 text-xs font-mono transition-colors block
                            ${m === selectedModel
                              ? "text-cyan-400 bg-cyan-400/10"
                              : "text-slate-300 hover:bg-white/5 hover:text-white"
                            }`}
                        >
                          {m}
                        </button>
                      ))}
                    </div>
                  )}
                </div>

                <button
                  onClick={handleNewUpload}
                  className="shrink-0 text-xs text-cyan-400 border border-cyan-400/30 rounded-lg px-2.5 py-1 hover:bg-cyan-400/10 transition-colors"
                >
                  New upload
                </button>
              </header>

              {/* Messages + Input */}
              <div className="flex-1 min-h-0">
                <ChatWindow
                  messages={messages}
                  isLoading={isLoading}
                  statusText={statusText}
                  streamingText={streamingText}
                  streamingThink={streamingThink}
                  onSend={sendMessage}
                  onCancel={cancelStream}
                />
              </div>
            </main>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
