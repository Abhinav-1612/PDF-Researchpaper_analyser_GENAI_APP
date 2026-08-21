import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { User, Bot, Clock, Database, Zap, ChevronRight } from "lucide-react";
import ThinkingAnimation from "./ThinkingAnimation";
import SourceFragments from "./SourceFragments";

/* ── Parse <think> block out of the raw model output ─────────────────────── */
function parseThinkContent(content) {
  const match = content.match(/<think>([\s\S]*?)(?:<\/think>|$)/);
  if (!match) return { think: null, answer: content };
  return {
    think: match[1].trim(),
    answer: content.replace(/<think>[\s\S]*?(?:<\/think>|$)/, "").trim(),
  };
}

/* ── Detect if streaming text is still inside a <think> block ───────────── */
function isInsideThink(text) {
  return text.trimStart().startsWith("<think>") && !text.includes("</think>");
}

/* ── Timing bar ─────────────────────────────────────────────────────────── */
function TimingBar({ timing }) {
  if (!timing) return null;
  const fmt = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`);
  return (
    <div className="flex items-center gap-3 mt-2 pt-2 border-t border-white/5 flex-wrap">
      <div className="flex items-center gap-1 text-[10px] font-mono text-slate-500">
        <Clock size={10} className="text-cyan-400/60" />
        <span className="text-slate-400">Total:</span>
        <span className="text-cyan-400">{fmt(timing.total_ms)}</span>
      </div>
      <div className="flex items-center gap-1 text-[10px] font-mono text-slate-500">
        <Database size={10} className="text-purple-400/60" />
        <span className="text-slate-400">Retrieval:</span>
        <span className="text-purple-400">{fmt(timing.retrieval_ms)}</span>
      </div>
      <div className="flex items-center gap-1 text-[10px] font-mono text-slate-500">
        <Zap size={10} className="text-orange-400/60" />
        <span className="text-slate-400">Stream:</span>
        <span className="text-orange-400">{fmt(timing.stream_ms)}</span>
      </div>
    </div>
  );
}

/* ── Collapsible think block ─────────────────────────────────────────────── */
function ThinkBlock({ think }) {
  const [open, setOpen] = useState(false);
  if (!think) return null;
  const wordCount = think.split(/\s+/).length;
  return (
    <div className="mb-3">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1.5 text-[11px] font-mono text-green-400/60 hover:text-green-400 transition-colors select-none"
      >
        <ChevronRight
          size={11}
          className={`transition-transform duration-200 ${open ? "rotate-90" : ""}`}
        />
        Chain of thought
        <span className="text-slate-600 ml-1">({wordCount} words)</span>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="mt-2 p-3 rounded-lg border border-green-400/10 bg-green-900/10 text-[11px] font-mono text-green-300/70 leading-relaxed whitespace-pre-wrap max-h-64 overflow-y-auto">
              {think}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

/* ── Main ChatMessage component ──────────────────────────────────────────── */
export default function ChatMessage({ msg, isStreaming, streamingText, statusText }) {
  const isUser = msg?.role === "user";
  const rawContent = isStreaming ? streamingText : (msg?.content || "");

  // While model is still inside <think> block, show ThinkingAnimation, not raw text
  const thinkStreaming = !isUser && isStreaming && isInsideThink(rawContent);

  // Prefer pre-parsed think from backend (msg.think). Fallback to live parsing if needed.
  const { think: parsedThink, answer } = parseThinkContent(rawContent);
  const think = msg?.think || parsedThink;
  const displayText = isUser ? rawContent : (answer || rawContent);
  
  if (!isUser) {
    console.log("ChatMessage render:", { msgId: msg?.id, msgThinkLength: msg?.think?.length, parsedThinkLength: parsedThink?.length, finalThinkLength: think?.length, isStreaming });
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className={`flex gap-3 ${isUser ? "flex-row-reverse" : "flex-row"}`}
    >
      {/* Avatar */}
      <div
        className={`shrink-0 w-8 h-8 rounded-full flex items-center justify-center border
          ${isUser
            ? "bg-cyan-400/10 border-cyan-400/30"
            : "bg-purple-500/10 border-purple-500/30"
          }`}
      >
        {isUser
          ? <User size={14} className="text-cyan-400" />
          : <Bot  size={14} className="text-purple-400" />
        }
      </div>

      {/* Bubble + metadata */}
      <div className={`max-w-[80%] min-w-0 flex flex-col gap-1 ${isUser ? "items-end" : "items-start"}`}>
        <div className={`rounded-2xl px-4 py-3 text-sm leading-relaxed w-full
          ${isUser ? "msg-user rounded-tr-sm" : "msg-assistant rounded-tl-sm"}`}>

          {/* Waiting for first token */}
          {!isUser && isStreaming && !streamingText && (
            <ThinkingAnimation statusText={statusText} />
          )}

          {/* Still inside <think> block — show pulsing animation, hide raw XML */}
          {thinkStreaming && (
            <ThinkingAnimation statusText="Thinking..." />
          )}

          {/* Collapsed think block */}
          {!isUser && think && (
            <ThinkBlock think={think} />
          )}

          {/* Main answer */}
          {!thinkStreaming && displayText && (
            isStreaming && streamingText
              /* During streaming — plain text + cursor, no markdown flicker */
              ? <p className="whitespace-pre-wrap text-slate-100 typing-cursor">
                  {displayText}
                </p>
              /* After streaming — full rendered markdown */
              : <div className="prose text-sm">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {displayText}
                  </ReactMarkdown>
                </div>
          )}

          {/* Timing bar */}
          {!isUser && !isStreaming && msg?.timing && (
            <TimingBar timing={msg.timing} />
          )}
        </div>

        {/* Fragments dropdown */}
        {!isUser && !isStreaming && msg?.sources?.length > 0 && (
          <div className="w-full">
            <SourceFragments sources={msg.sources} />
          </div>
        )}
      </div>
    </motion.div>
  );
}
