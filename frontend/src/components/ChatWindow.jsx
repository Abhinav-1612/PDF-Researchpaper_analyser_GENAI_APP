import { useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Send, Square, MessageSquare } from "lucide-react";
import ChatMessage from "./ChatMessage";
import { useState } from "react";

export default function ChatWindow({ messages, isLoading, statusText, streamingText, onSend, onCancel }) {
  const [input, setInput] = useState("");
  const bottomRef = useRef(null);
  const textareaRef = useRef(null);

  // Auto-scroll on new content
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingText]);

  const handleSend = () => {
    if (!input.trim() || isLoading) return;
    onSend(input.trim());
    setInput("");
    textareaRef.current?.focus();
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-5">
        {messages.length === 0 && !isLoading && (
          <div className="h-full flex flex-col items-center justify-center text-center opacity-50 select-none">
            <MessageSquare size={40} className="text-cyan-400/40 mb-4" />
            <p className="text-slate-400 text-sm">Documents indexed. Ask anything about them.</p>
          </div>
        )}

        <AnimatePresence>
          {messages.map((msg) => (
            <ChatMessage key={msg.id} msg={msg} isStreaming={false} />
          ))}
        </AnimatePresence>

        {/* Streaming placeholder bubble */}
        {isLoading && (
          <ChatMessage
            msg={{ role: "assistant" }}
            isStreaming
            streamingText={streamingText}
            statusText={statusText}
          />
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <div className="shrink-0 px-4 pb-4">
        <div className="glass rounded-2xl flex items-end gap-2 p-2 pl-4 border border-white/8">
          <textarea
            ref={textareaRef}
            rows={1}
            value={input}
            onChange={(e) => {
              setInput(e.target.value);
              e.target.style.height = "auto";
              e.target.style.height = Math.min(e.target.scrollHeight, 160) + "px";
            }}
            onKeyDown={handleKey}
            placeholder="Ask a question about your documents…"
            disabled={isLoading}
            className="flex-1 bg-transparent text-sm text-slate-200 placeholder-slate-500 resize-none outline-none py-1.5 leading-relaxed max-h-40 overflow-y-auto"
          />

          <div className="flex gap-1.5 shrink-0 pb-0.5">
            {isLoading ? (
              <button
                onClick={onCancel}
                className="w-8 h-8 rounded-xl bg-red-500/20 border border-red-500/40 text-red-400 flex items-center justify-center hover:bg-red-500/30 transition-all"
                title="Cancel"
              >
                <Square size={13} />
              </button>
            ) : (
              <motion.button
                whileTap={{ scale: 0.92 }}
                onClick={handleSend}
                disabled={!input.trim()}
                className="w-8 h-8 rounded-xl btn-neon flex items-center justify-center disabled:opacity-30"
                title="Send (Enter)"
              >
                <Send size={13} />
              </motion.button>
            )}
          </div>
        </div>
        <p className="text-[10px] text-slate-600 mt-1 ml-2">
          Enter to send · Shift+Enter for newline
        </p>
      </div>
    </div>
  );
}
