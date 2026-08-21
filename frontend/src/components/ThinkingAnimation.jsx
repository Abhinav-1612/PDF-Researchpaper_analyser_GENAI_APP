export default function ThinkingAnimation({ statusText }) {
  return (
    <div className="flex items-center gap-3 py-2">
      <div className="flex gap-1.5">
        <div className="thinking-dot" />
        <div className="thinking-dot" />
        <div className="thinking-dot" />
      </div>
      {statusText && (
        <span className="text-xs text-cyan-400/80 font-mono tracking-wide animate-pulse">
          {statusText}
        </span>
      )}
    </div>
  );
}
