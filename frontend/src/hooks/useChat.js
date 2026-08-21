import { useState, useCallback, useRef } from "react";
import { streamQuery } from "../api/client";

export function useChat(selectedModel) {
  const [messages, setMessages]     = useState([]);
  const [isLoading, setIsLoading]   = useState(false);
  const [statusText, setStatusText] = useState("");
  const [streamingText, setStreamingText] = useState("");
  const controllerRef = useRef(null);
  const timingRef = useRef(null); // holds {retrieval_ms, stream_ms, total_ms}

  const sendMessage = useCallback(
    (query) => {
      if (!query.trim() || isLoading) return;

      // Build plain chat history for the API
      const chatHistory = messages.map((m) => ({
        role: m.role,
        content: m.content,
      }));

      // Optimistically add the user message
      const userMsg = { id: Date.now(), role: "user", content: query };
      setMessages((prev) => [...prev, userMsg]);

      setIsLoading(true);
      setStatusText("Connecting...");
      setStreamingText("");
      timingRef.current = null;

      let fullAnswer = "";
      let sources    = [];
      let timing     = null;
      let thinkContent = null;

      controllerRef.current = streamQuery({
        query,
        model: selectedModel,
        chatHistory,
        onStatus: (s) => setStatusText(s),
        onToken: (t) => {
          fullAnswer += t;
          setStreamingText(fullAnswer);
        },
        onThinking: (t) => { thinkContent = t; },
        onSources: (s) => { sources = s; },
        onTiming:  (t) => { timing = t; },
        onDone: () => {
          setMessages((prev) => [
            ...prev,
            {
              id: Date.now() + 1,
              role: "assistant",
              content: fullAnswer,
              think: thinkContent,
              sources,
              timing,
            },
          ]);
          setIsLoading(false);
          setStatusText("");
          setStreamingText("");
          timingRef.current = null;
        },
        onError: (err) => {
          setMessages((prev) => [
            ...prev,
            {
              id: Date.now() + 1,
              role: "assistant",
              content: `⚠️ Error: ${err}`,
              sources: [],
              isError: true,
            },
          ]);
          setIsLoading(false);
          setStatusText("");
          setStreamingText("");
        },
      });
    },
    [messages, isLoading, selectedModel]
  );

  const cancelStream = useCallback(() => {
    controllerRef.current?.abort();
    setIsLoading(false);
    setStatusText("");
  }, []);

  const clearChat = useCallback(() => {
    setMessages([]);
    setStreamingText("");
    setStatusText("");
  }, []);

  return { messages, isLoading, statusText, streamingText, sendMessage, cancelStream, clearChat };
}
