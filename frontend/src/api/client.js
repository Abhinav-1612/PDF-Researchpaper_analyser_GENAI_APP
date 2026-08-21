import axios from "axios";

const BASE_URL = "http://localhost:8000";

const api = axios.create({ baseURL: BASE_URL });

export const healthCheck = () => api.get("/health");
export const getModels   = () => api.get("/models");

export const uploadFiles = (files) => {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  return api.post("/upload", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
};

/**
 * Opens an SSE connection to /stream.
 * Calls onStatus, onToken, onSources, onDone, onError callbacks.
 * Returns an AbortController to cancel the stream.
 */
export function streamQuery({ query, model, chatHistory = [], onStatus, onToken, onSources, onTiming, onDone, onError }) {
  const controller = new AbortController();

  fetch(`${BASE_URL}/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, model, chat_history: chatHistory }),
    signal: controller.signal,
  })
    .then(async (res) => {
      if (!res.ok) {
        const err = await res.text();
        onError?.(err);
        return;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop(); // keep incomplete chunk

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const event = JSON.parse(line.slice(6));
            if (event.type === "status")  onStatus?.(event.content);
            if (event.type === "token")   onToken?.(event.content);
            if (event.type === "sources") onSources?.(event.content);
            if (event.type === "timing")  onTiming?.(event.content);
            if (event.type === "done")    onDone?.();
            if (event.type === "error")   onError?.(event.content);
          } catch {}
        }
      }
    })
    .catch((err) => {
      if (err.name !== "AbortError") onError?.(err.message);
    });

  return controller;
}

export default api;
