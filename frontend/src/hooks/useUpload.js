import { useState, useCallback } from "react";
import { uploadFiles, getModels } from "../api/client";

export function useUpload() {
  const [files, setFiles]         = useState([]);
  const [status, setStatus]       = useState("idle"); // idle | uploading | done | error
  const [progress, setProgress]   = useState(0);
  const [stats, setStats]         = useState(null);
  const [error, setError]         = useState(null);
  const [models, setModels]       = useState([]);
  const [defaultModel, setDefaultModel] = useState("");

  const loadModels = useCallback(async () => {
    try {
      const { data } = await getModels();
      setModels(data.models || []);
      setDefaultModel(data.default || "");
    } catch {}
  }, []);

  const addFiles = useCallback((newFiles) => {
    setFiles((prev) => {
      const names = new Set(prev.map((f) => f.name));
      const filtered = newFiles.filter((f) => !names.has(f.name));
      return [...prev, ...filtered];
    });
  }, []);

  const removeFile = useCallback((name) => {
    setFiles((prev) => prev.filter((f) => f.name !== name));
  }, []);

  const clearAll = useCallback(() => {
    setFiles([]);
    setStatus("idle");
    setStats(null);
    setError(null);
  }, []);

  const upload = useCallback(async () => {
    if (!files.length) return;
    setStatus("uploading");
    setError(null);
    setProgress(10);

    try {
      const fakeProgress = setInterval(() => {
        setProgress((p) => Math.min(p + Math.random() * 15, 85));
      }, 600);

      const { data } = await uploadFiles(files);
      clearInterval(fakeProgress);
      setProgress(100);
      setStats(data);
      setStatus("done");
    } catch (e) {
      setError(e.response?.data?.detail || e.message || "Upload failed");
      setStatus("error");
    }
  }, [files]);

  return { files, addFiles, removeFile, clearAll, upload, status, progress, stats, error, models, defaultModel, loadModels };
}
