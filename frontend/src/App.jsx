import React, { useState, useRef, useEffect, useMemo } from "react";
import ErrorBoundary from "./ErrorBoundary";
import ConnectionGraph from "./ConnectionGraph";

const llmOptions = [
  { label: "OpenAI (GPT-4.1)", value: "openai" },
  { label: "Gemini 2.5", value: "gemini" },
  { label: "Ollama (LLaMA)", value: "ollama" },
];

const searchOptions = [
  { label: "Tavily (Default)", value: "tavily" },
  { label: "SerpApi", value: "serpapi" },
  { label: "OpenAI Web Search", value: "openai_websearch" },
  { label: "Gemini Web Search", value: "gemini_websearch" },
  { label: "Selenium (Fallback)", value: "selenium" },
];

const sampleSubject = "Aleena Farrow";
const sampleObjectives = [
  "Surface hidden ownership structures",
  "Identify sanction exposure",
  "Map suspicious connections",
];

function tryParseJSONSafe(str) {
  if (!str || typeof str !== "string") return null;
  const trimmed = str.trim();
  if (!(trimmed.startsWith("{") || trimmed.startsWith("["))) return null;
  try {
    return JSON.parse(trimmed);
  } catch {
    return null;
  }
}

function extractQueriesFromLogEntry(entry) {
  const found = [];
  if (!entry) return found;

  const asObj = tryParseJSONSafe(entry);
  if (asObj) {
    if (Array.isArray(asObj)) {
      asObj.forEach((x) => {
        if (x && typeof x === "object") {
          if (x.query) found.push(String(x.query));
          if (x.prompt) found.push(String(x.prompt));
          if (x.input) found.push(String(x.input));
        }
      });
    } else if (typeof asObj === "object") {
      if (asObj.query) found.push(String(asObj.query));
      if (asObj.prompt) found.push(String(asObj.prompt));
      if (asObj.inputs) {
        if (Array.isArray(asObj.inputs))
          asObj.inputs.forEach((i) => found.push(String(i)));
        else found.push(String(asObj.inputs));
      }
    }
  } else {
    const lower = String(entry).toLowerCase();
    const regexes = [
      /query[:=]\s*["']?([^"'\n]+)/i,
      /prompt[:=]\s*["']?([^"'\n]+)/i,
      /search[:=]\s*["']?([^"'\n]+)/i,
      /issued query\s*["']([^"']+)["']/i,
      /running search for\s*["']([^"']+)["']/i,
      /refined query\s*[:\-]\s*(.+)$/i,
    ];
    for (const r of regexes) {
      const m = entry.match(r);
      if (m && m[1]) {
        found.push(m[1].trim());
      }
    }

    if (
      entry.length > 10 &&
      entry.length < 200 &&
      !lower.includes("http") &&
      /[a-zA-Z]/.test(entry) &&
      (lower.includes("investigation") ||
        lower.includes("query") ||
        lower.includes("search") ||
        entry.split(" ").length > 3)
    ) {
      if (
        /^(search|query|prompt|refined|asking)/i.test(entry) ||
        /[?]/.test(entry)
      ) {
        found.push(entry.trim());
      }
    }
  }
  return found;
}

function extractTelemetryFromLogEntry(entry) {
  if (!entry) return null;

  const asObj = tryParseJSONSafe(entry);
  if (asObj && typeof asObj === "object") {
    if (asObj.llm && asObj.usage) {
      return {
        type: "token_usage",
        provider: asObj.llm || asObj.provider,
        input_tokens: Number(
          asObj.usage.prompt_tokens || asObj.usage.input_tokens || 0,
        ),
        output_tokens: Number(
          asObj.usage.completion_tokens || asObj.usage.output_tokens || 0,
        ),
        raw: asObj,
      };
    }
    if (
      asObj.event === "llm.summary" ||
      asObj.event === "llm_call_summary" ||
      asObj.event === "llm.summary"
    ) {
      return {
        type: "llm_summary",
        provider: asObj.provider || asObj.llm,
        ttft: asObj.ttft ?? asObj.TTFT ?? null,
        end_to_end: asObj.end_to_end ?? asObj.e2e ?? null,
        total_tokens: asObj.total_tokens ?? asObj.tokens ?? null,
        raw: asObj,
      };
    }
    if (asObj.metric && asObj.value !== undefined) {
      return {
        type: "metric",
        name: asObj.metric,
        value: asObj.value,
        tags: asObj.tags,
      };
    }
  } else {
    const lower = String(entry).toLowerCase();
    const tokenMatch = lower.match(
      /(?:input_tokens|prompt_tokens)[=:\s]+(\d+)/,
    );
    const outputMatch = lower.match(
      /(?:output_tokens|completion_tokens|completion_tokens)[=:\s]+(\d+)/,
    );
    const totalMatch = lower.match(/(?:total_tokens|tokens)[=:\s]+(\d+)/);
    const ttftMatch = lower.match(
      /(?:ttft|time to first token)[=:\s]+([\d.]+)/i,
    );
    const e2eMatch = lower.match(
      /(?:end[-\s]*to[-\s]*end|e2e|end_to_end)[=:\s]+([\d.]+)/i,
    );

    if (tokenMatch || outputMatch || totalMatch || ttftMatch || e2eMatch) {
      return {
        type: "telemetry_simple",
        input_tokens: tokenMatch ? parseInt(tokenMatch[1], 10) : 0,
        output_tokens: outputMatch ? parseInt(outputMatch[1], 10) : 0,
        total_tokens: totalMatch
          ? parseInt(totalMatch[1], 10)
          : tokenMatch || outputMatch
            ? Number(tokenMatch ? tokenMatch[1] : 0) +
              Number(outputMatch ? outputMatch[1] : 0)
            : 0,
        ttft: ttftMatch ? parseFloat(ttftMatch[1]) : null,
        end_to_end: e2eMatch ? parseFloat(e2eMatch[1]) : null,
        raw: entry,
      };
    }
  }
  return null;
}

function dedupeKeepLatest(arr) {
  const seen = new Set();
  const out = [];
  for (let i = arr.length - 1; i >= 0; i--) {
    const s = arr[i];
    if (!seen.has(s)) {
      seen.add(s);
      out.push(s);
    }
  }
  return out.reverse();
}

function App() {
  const [subject, setSubject] = useState(sampleSubject);
  const [objectives, setObjectives] = useState(sampleObjectives.join("\n"));
  const [selectedLlms, setSelectedLlms] = useState(["openai", "gemini"]);
  const [selectedSearchProvider, setSelectedSearchProvider] =
    useState("tavily");

  const [statusLogs, setStatusLogs] = useState([]);
  const [results, setResults] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const [liveQueries, setLiveQueries] = useState([]);
  const [telemetryEvents, setTelemetryEvents] = useState([]);
  const [showRawLogs, setShowRawLogs] = useState(false);

  const wsRef = useRef(null);

  useEffect(() => {
    const queries = [];
    const telemetry = [];

    for (const entry of statusLogs) {
      try {
        const parsedQueries = extractQueriesFromLogEntry(entry);
        queries.push(...parsedQueries);

        const t = extractTelemetryFromLogEntry(entry);
        if (t) telemetry.push(t);
      } catch {
        // ignore parse errors
      }
    }

    if (results && results.refined_queries) {
      results.refined_queries.forEach((q) => queries.push(q));
    }

    setLiveQueries((prev) =>
      dedupeKeepLatest([...prev, ...queries]).slice(-50),
    );
    setTelemetryEvents((prev) => {
      const merged = [...prev, ...telemetry];
      return merged.length > 200 ? merged.slice(merged.length - 200) : merged;
    });
  }, [statusLogs, results]);

  function toggleLlmSelection(value) {
    setSelectedLlms((prev) =>
      prev.includes(value)
        ? prev.filter((id) => id !== value)
        : [...prev, value],
    );
  }

  function clearLiveData() {
    setStatusLogs([]);
    setResults(null);
    setErrorMessage("");
    setLiveQueries([]);
    setTelemetryEvents([]);
  }

  async function startInvestigation() {
    if (isRunning) return;

    clearLiveData();

    setIsRunning(true);

    try {
      const resp = await fetch("http://127.0.0.1:8000/api/investigate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          subject: subject.trim(),
          objectives: objectives
            .split("\n")
            .map((s) => s.trim())
            .filter((s) => s.length > 0),
          llms: selectedLlms,
          search_provider: selectedSearchProvider,
        }),
      });

      if (!resp.ok) {
        const err = await resp.text();
        setErrorMessage(`API Error: ${resp.status} ${err}`);
        setIsRunning(false);
        return;
      }

      const data = await resp.json();
      const jobId = data.job_id;

      if (!jobId) {
        setErrorMessage("Failed to get job ID from server");
        setIsRunning(false);
        return;
      }

      const ws = new WebSocket(`ws://127.0.0.1:8000/ws/status/${jobId}`);
      wsRef.current = ws;

      ws.onopen = () => {
        setStatusLogs((logs) => [...logs, "Connected to backend for updates"]);
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.error) {
            setErrorMessage(msg.error);
            ws.close();
            setIsRunning(false);
            return;
          }

          if (msg.status) {
            const incomingLogs = Array.isArray(msg.logs)
              ? msg.logs.map((l) =>
                  typeof l === "string" ? l : JSON.stringify(l),
                )
              : [];
            setStatusLogs((logs) => {
              const merged = [...logs, ...incomingLogs];
              return merged.slice(-1000);
            });

            if (msg.results) {
              setResults(msg.results);
              if (msg.results.refined_queries) {
                setLiveQueries((prev) =>
                  dedupeKeepLatest([
                    ...prev,
                    ...msg.results.refined_queries.map((q) => String(q)),
                  ]).slice(-50),
                );
              }
            }
          }

          if (msg.status === "Completed") {
            setStatusLogs((logs) => [...logs, "Investigation completed"]);
            setIsRunning(false);
            ws.close();
            wsRef.current = null;
          }
        } catch {
          setStatusLogs((logs) => [...logs, String(event.data)].slice(-1000));
          setErrorMessage("Malformed message from server");
          setIsRunning(false);
          if (ws) ws.close();
          wsRef.current = null;
        }
      };

      ws.onerror = () => {
        setErrorMessage("WebSocket connection error");
        setIsRunning(false);
      };

      ws.onclose = () => {
        setStatusLogs((logs) => [...logs, "WebSocket connection closed"]);
      };
    } catch (e) {
      setErrorMessage(`Connection failed: ${e.message}`);
      setIsRunning(false);
    }
  }

  const telemetrySummary = useMemo(() => {
    let totalInput = 0;
    let totalOutput = 0;
    let totalT = 0;
    let ttftCount = 0;
    let calls = 0;

    for (const t of telemetryEvents) {
      if (!t) continue;
      if (["token_usage", "telemetry_simple", "llm_summary"].includes(t.type)) {
        const inT = Number(t.input_tokens || 0);
        const outT = Number(t.output_tokens || 0);
        const tot = Number(t.total_tokens || inT + outT || 0);
        totalInput += inT;
        totalOutput += outT;
        calls += 1;
      }
      if (t.ttft) {
        totalT += Number(t.ttft);
        ttftCount += 1;
      }
    }

    const avgTTFT = ttftCount ? totalT / ttftCount : null;
    return {
      calls,
      totalInput,
      totalOutput,
      totalTokens: totalInput + totalOutput,
      avgTTFT,
    };
  }, [telemetryEvents]);

  const featureCoverage = useMemo(() => {
    const hasFacts =
      results && Array.isArray(results.facts) && results.facts.length > 0;
    const hasRisks =
      results && Array.isArray(results.risks) && results.risks.length > 0;
    const hasConnections =
      results &&
      Array.isArray(results.connections) &&
      results.connections.length > 0;
    const hasReports =
      results &&
      results.reports &&
      (results.reports.json || results.reports.markdown);
    const refined =
      liveQueries.length > 3 ||
      statusLogs.some((l) => /refin(ed|ement)|refine/i.test(l));
    return {
      deepFactExtraction: hasFacts,
      riskPatternRecognition: hasRisks,
      connectionMapping: hasConnections,
      sourceValidation: hasReports,
      dynamicQueryRefinement: refined,
    };
  }, [results, statusLogs, liveQueries]);

  return (
    <ErrorBoundary>
      <div
        style={{
          fontFamily: "Inter, Arial, sans-serif",
          padding: 28,
          margin: "12px auto",
          display: "grid",
          gridTemplateColumns: "300px 2fr 1.5fr",
          gap: 24,
          height: "100vh",
          overflow: "hidden",
        }}
      >
        {/* Left: Inputs / Controls */}
        <div
          style={{
            borderRight: "1px solid #e6e9ee",
            paddingRight: 20,
            overflowY: "auto",
            maxHeight: "100vh",
          }}
        >
          <h1 style={{ marginTop: 4, marginBottom: 8 }}>
            Deep Research AI Agent
          </h1>
          <p style={{ color: "#666", marginTop: 0, marginBottom: 12 }}>
            Live demo - enhanced telemetry & query panels
          </p>

          <section style={{ marginBottom: 18 }}>
            <h3 style={{ marginBottom: 8 }}>Configuration</h3>
            <div style={{ marginBottom: 10 }}>
              <strong>LLMs:</strong>
              <div style={{ marginTop: 8 }}>
                {llmOptions.map((opt) => (
                  <label
                    key={opt.value}
                    style={{ display: "block", marginBottom: 6 }}
                  >
                    <input
                      type="checkbox"
                      value={opt.value}
                      checked={selectedLlms.includes(opt.value)}
                      onChange={() => toggleLlmSelection(opt.value)}
                      disabled={
                        selectedLlms.length === 1 &&
                        selectedLlms.includes(opt.value)
                      }
                    />{" "}
                    {opt.label}
                  </label>
                ))}
                <small style={{ color: "#666" }}>
                  Select at least two distinct models. Fallback to LLaMA if
                  others fail.
                </small>
              </div>
            </div>

            <div style={{ marginTop: 10, marginBottom: 10 }}>
              <strong>Search Provider:</strong>
              <br />
              <select
                value={selectedSearchProvider}
                onChange={(e) => setSelectedSearchProvider(e.target.value)}
                style={{ width: "100%", padding: 8, marginTop: 8 }}
              >
                {searchOptions.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
              <small style={{ color: "#666" }}>
                Default is Tavily. Fallbacks available.
              </small>
            </div>
          </section>

          <section style={{ marginBottom: 16 }}>
            <h3 style={{ marginBottom: 8 }}>Input</h3>
            <label style={{ display: "block", marginBottom: 8 }}>
              Subject:
              <input
                type="text"
                value={subject}
                onChange={(e) => setSubject(e.target.value)}
                style={{
                  width: "100%",
                  padding: 8,
                  fontSize: 14,
                  marginTop: 6,
                }}
                placeholder="Full name or organization"
              />
            </label>

            <label style={{ display: "block", marginBottom: 6 }}>
              Objectives (one per line):
              <textarea
                rows={5}
                value={objectives}
                onChange={(e) => setObjectives(e.target.value)}
                style={{
                  width: "100%",
                  padding: 8,
                  fontSize: 14,
                  marginTop: 6,
                }}
              />
            </label>

            <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
              <button
                onClick={startInvestigation}
                disabled={isRunning}
                style={{
                  flex: 1,
                  padding: "10px 16px",
                  fontSize: 15,
                  backgroundColor: isRunning ? "#9aa4b2" : "#0f62fe",
                  color: "white",
                  border: "none",
                  borderRadius: 6,
                  cursor: isRunning ? "default" : "pointer",
                }}
              >
                {isRunning ? "Running..." : "Run Investigation"}
              </button>
              <button
                onClick={clearLiveData}
                style={{
                  padding: "10px 12px",
                  fontSize: 14,
                  backgroundColor: "#f3f4f6",
                  color: "#111827",
                  border: "1px solid #e5e7eb",
                  borderRadius: 6,
                }}
              >
                Clear
              </button>
            </div>

            {errorMessage && (
              <p style={{ color: "crimson", marginTop: 10 }}>
                <strong>Error:</strong> {errorMessage}
              </p>
            )}
          </section>

          <section style={{ marginTop: 6 }}>
            <h3 style={{ marginBottom: 8 }}>
              Functional Specifications (live)
            </h3>
            <ul style={{ paddingLeft: 18, color: "#333", maxWidth: 400 }}>
              <li>
                <input
                  type="checkbox"
                  readOnly
                  checked={featureCoverage.deepFactExtraction}
                />{" "}
                Deep Fact Extraction
              </li>
              <li>
                <input
                  type="checkbox"
                  readOnly
                  checked={featureCoverage.riskPatternRecognition}
                />{" "}
                Risk Pattern Recognition
              </li>
              <li>
                <input
                  type="checkbox"
                  readOnly
                  checked={featureCoverage.connectionMapping}
                />{" "}
                Connection Mapping
              </li>
              <li>
                <input
                  type="checkbox"
                  readOnly
                  checked={featureCoverage.sourceValidation}
                />{" "}
                Source Validation & Reports
              </li>
              <li>
                <input
                  type="checkbox"
                  readOnly
                  checked={featureCoverage.dynamicQueryRefinement}
                />{" "}
                Dynamic Query Refinement
              </li>
            </ul>
          </section>
        </div>

        {/* Center: Status / Results */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            overflowY: "auto",
          }}
        >
          <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
            <h2 style={{ margin: 0 }}>Status & Results</h2>
            <div style={{ marginLeft: "auto", color: "#666" }}>
              Job status:{" "}
              {isRunning ? (
                <strong style={{ color: "#0f62fe" }}>Running</strong>
              ) : (
                <strong>Idle</strong>
              )}
            </div>
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 14,
              marginTop: 16,
              alignItems: "start",
            }}
          >
            {/* Status logs */}
            <div
              style={{
                border: "1px solid #e6e9ee",
                borderRadius: 8,
                padding: 12,
                minHeight: 240,
                background: "#fff",
                overflowY: "auto",
              }}
            >
              <h4 style={{ marginTop: 0 }}>Activity Logs</h4>
              <div
                style={{
                  fontFamily: "monospace",
                  fontSize: 13,
                  color: "#111827",
                  maxHeight: 320,
                  overflowY: "auto",
                  whiteSpace: "pre-wrap",
                }}
              >
                {statusLogs.length === 0 ? (
                  <em style={{ color: "#6b7280" }}>
                    No logs yet. Run an investigation to see live activity.
                  </em>
                ) : (
                  statusLogs.slice(-500).map((log, idx) => {
                    const l = String(log);
                    const isQuery = /query|prompt|search|refin/i.test(l);
                    const isWarn = /warn|error|failed|exception|timeout/i.test(
                      l,
                    );
                    return (
                      <div
                        key={idx}
                        style={{
                          padding: "4px 0",
                          borderBottom: "1px dashed #f1f5f9",
                        }}
                      >
                        <span
                          style={{
                            color: isWarn
                              ? "#b91c1c"
                              : isQuery
                                ? "#0f62fe"
                                : "#374151",
                          }}
                        >
                          &gt;{" "}
                        </span>
                        <span style={{ marginLeft: 6 }}>{l}</span>
                      </div>
                    );
                  })
                )}
              </div>
              <div style={{ marginTop: 8, display: "flex", gap: 8 }}>
                <button
                  onClick={() => setShowRawLogs((s) => !s)}
                  style={{
                    padding: "6px 8px",
                    borderRadius: 6,
                    border: "1px solid #e5e7eb",
                    background: "#fff",
                  }}
                >
                  {showRawLogs ? "Hide Raw" : "Show Raw"}
                </button>
                <button
                  onClick={() => {
                    const blob = new Blob([statusLogs.join("\n")], {
                      type: "text/plain;charset=utf-8",
                    });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement("a");
                    a.href = url;
                    a.download = `${subject.replace(/\s+/g, "_")}_logs.txt`;
                    a.click();
                    URL.revokeObjectURL(url);
                  }}
                  style={{
                    padding: "6px 8px",
                    borderRadius: 6,
                    border: "1px solid #e5e7eb",
                    background: "#fff",
                  }}
                >
                  Download Logs
                </button>
              </div>
            </div>

            {/* Live Queries */}
            <div
              style={{
                border: "1px solid #e6e9ee",
                borderRadius: 8,
                padding: 12,
                minHeight: 240,
                background: "#fff",
                overflowY: "auto",
              }}
            >
              <h4 style={{ marginTop: 0 }}>Live Queries / Prompts</h4>
              <div
                style={{
                  fontSize: 14,
                  color: "#111827",
                  maxHeight: 320,
                  overflowY: "auto",
                  wordBreak: "break-word",
                }}
              >
                {liveQueries.length === 0 ? (
                  <em style={{ color: "#6b7280" }}>No queries detected yet.</em>
                ) : (
                  liveQueries
                    .slice(-50)
                    .reverse()
                    .map((q, i) => (
                      <div
                        key={i}
                        style={{
                          padding: 8,
                          borderBottom: "1px dashed #f1f5f9",
                        }}
                      >
                        <strong style={{ display: "block", color: "#0f172a" }}>
                          Query #{liveQueries.length - i}
                        </strong>
                        <div
                          style={{
                            marginTop: 6,
                            color: "#374151",
                            fontFamily: "monospace",
                            wordBreak: "break-word",
                          }}
                        >
                          {q}
                        </div>
                      </div>
                    ))
                )}
              </div>
            </div>
          </div>

          {/* Results / reports */}
          <div
            style={{
              marginTop: 14,
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 16,
            }}
          >
            <div
              style={{
                border: "1px solid #e6e9ee",
                borderRadius: 8,
                padding: 12,
                background: "#fff",
                minHeight: 280,
                overflowY: "auto",
              }}
            >
              <h4 style={{ marginTop: 0 }}>Results</h4>
              {results ? (
                <>
                  <h5 style={{ marginBottom: 6 }}>Facts</h5>
                  <ul
                    style={{
                      maxHeight: 130,
                      overflowY: "auto",
                      paddingLeft: 18,
                    }}
                  >
                    {Array.isArray(results.facts) &&
                    results.facts.length > 0 ? (
                      results.facts.map((f, i) => (
                        <li key={i} style={{ marginBottom: 6 }}>
                          {typeof f === "string"
                            ? f
                            : f.fact ||
                              f.normalized_fact ||
                              (f.title || JSON.stringify(f))
                                .toString()
                                .slice(0, 300)}
                        </li>
                      ))
                    ) : (
                      <li style={{ color: "#6b7280" }}>No facts found.</li>
                    )}
                  </ul>

                  <h5 style={{ marginBottom: 6, marginTop: 8 }}>Risks</h5>
                  <ul
                    style={{
                      maxHeight: 120,
                      overflowY: "auto",
                      paddingLeft: 18,
                    }}
                  >
                    {Array.isArray(results.risks) &&
                    results.risks.length > 0 ? (
                      results.risks.map((r, i) => (
                        <li key={i} style={{ marginBottom: 6 }}>
                          {typeof r === "string"
                            ? r
                            : r.label || r.description || JSON.stringify(r)}
                        </li>
                      ))
                    ) : (
                      <li style={{ color: "#6b7280" }}>No risks surfaced.</li>
                    )}
                  </ul>

                  <div style={{ marginTop: 8 }}>
                    <h5 style={{ marginBottom: 6 }}>Connections</h5>
                    <div style={{ maxHeight: 140, overflowY: "auto" }}>
                      {Array.isArray(results.connections) &&
                      results.connections.length > 0 ? (
                        <ul style={{ paddingLeft: 18 }}>
                          {results.connections.map((c, idx) => (
                            <li key={idx} style={{ marginBottom: 6 }}>
                              <strong>{c.source || c.from || "Unknown"}</strong>{" "}
                              → <strong>{c.target || c.to || "Unknown"}</strong>{" "}
                              ({c.relation || c.label || "associated_with"})
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <em style={{ color: "#6b7280" }}>
                          No connections found.
                        </em>
                      )}
                    </div>
                  </div>

                  <div style={{ marginTop: 8 }}>
                    <h5 style={{ marginBottom: 6 }}>Reports</h5>
                    <div>
                      {results.reports && results.reports.json ? (
                        <a
                          href={results.reports.json}
                          target="_blank"
                          rel="noreferrer"
                          style={{ marginRight: 12, color: "#0f62fe" }}
                        >
                          Download JSON
                        </a>
                      ) : (
                        <span style={{ color: "#6b7280" }}>No JSON report</span>
                      )}
                      {results.reports && results.reports.markdown ? (
                        <a
                          href={results.reports.markdown}
                          target="_blank"
                          rel="noreferrer"
                          style={{ marginLeft: 12, color: "#0f62fe" }}
                        >
                          Download Markdown
                        </a>
                      ) : (
                        <span style={{ marginLeft: 12, color: "#6b7280" }}>
                          No Markdown report
                        </span>
                      )}
                    </div>
                  </div>
                </>
              ) : (
                <div style={{ color: "#6b7280" }}>
                  No results yet. Click 'Run Investigation' to start.
                </div>
              )}
            </div>

            {/* Connection graph */}
            <div
              style={{
                border: "1px solid #e6e9ee",
                borderRadius: 8,
                padding: 12,
                background: "#fff",
                minHeight: 280,
                overflowY: "auto",
              }}
            >
              <h4 style={{ marginTop: 0 }}>Connection Graph (Preview)</h4>
              <div style={{ height: 280 }}>
                <ConnectionGraph
                  edges={(results && results.connections) || []}
                  nodesFromEdges
                />
              </div>

              <h4 style={{ marginTop: 12 }}>Telemetry</h4>
              <div style={{ height: 175, fontSize: 13, color: "#111827" }}>
                <div>LLM Calls</div>
                <div>{telemetrySummary.calls}</div>
                <div>Input Tokens</div>
                <div>{telemetrySummary.totalInput}</div>
                <div>Output Tokens</div>
                <div>{telemetrySummary.totalOutput}</div>
                <div>Total Tokens</div>
                <div>{telemetrySummary.totalTokens}</div>
                <div>Avg TTFT</div>
                <div>
                  {telemetrySummary.avgTTFT
                    ? telemetrySummary.avgTTFT.toFixed(3) + "s"
                    : "—"}
                </div>

                <h5 style={{ marginTop: 12, marginBottom: 6 }}>
                  Recent Telemetry Events
                </h5>
                <div
                  style={{
                    maxHeight: 120,
                    overflowY: "auto",
                    fontFamily: "monospace",
                    fontSize: 12,
                    color: "#374151",
                  }}
                >
                  {telemetryEvents.length === 0 ? (
                    <div style={{ color: "#6b7280" }}>No telemetry yet.</div>
                  ) : (
                    telemetryEvents
                      .slice(-80)
                      .reverse()
                      .map((t, i) => (
                        <div
                          key={i}
                          style={{
                            padding: "6px 4px",
                            borderBottom: "1px dashed #f1f5f9",
                          }}
                        >
                          <div>
                            <strong>{t.type || "evt"}</strong>{" "}
                            {t.provider ? `@${t.provider}` : ""}
                          </div>
                          <div style={{ marginTop: 4, color: "#4b5563" }}>
                            {t.raw
                              ? typeof t.raw === "string"
                                ? t.raw
                                : JSON.stringify(t.raw)
                              : JSON.stringify(t)}
                          </div>
                        </div>
                      ))
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </ErrorBoundary>
  );
}

export default App;
