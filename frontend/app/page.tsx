"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  Activity,
  AlertTriangle,
  CheckCircle,
  ShieldAlert,
  WifiOff,
} from "lucide-react";

const WS_URL =
  process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws/telemetry";
const HEALTH_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "";

const MAX_LOGS = 50;
const MAX_CHART_POINTS = 60;
const HEALTH_POLL_MS = 10_000;
// A node with no update for this long is shown as stale rather than silently
// displaying its last known value as though it were current.
const STALE_AFTER_MS = 30_000;

const RECONNECT_BASE_MS = 1_000;
const RECONNECT_MAX_MS = 30_000;

// Distinct hues per microphone. Previously every node was drawn as one shared
// "probability" series, so four machines interleaved into a single meaningless
// zig-zag and an operator could not tell which one was failing.
const NODE_COLORS = [
  "#10B981",
  "#3B82F6",
  "#F59E0B",
  "#EC4899",
  "#8B5CF6",
  "#14B8A6",
];

type Severity = "normal" | "warning" | "critical";
type ConnectionStatus = "connected" | "connecting" | "disconnected";

type TelemetryFrame = {
  node_id: number;
  timestamp: number;
  telemetry: string;
  anomaly: {
    score: number;
    severity: Severity;
    is_anomaly: boolean;
    z_score: number;
  };
  ttf_prediction: number;
  generated: boolean;
};

type TelemetryLog = {
  id: string;
  time: string;
  severity: Severity;
  text: string;
  nodeId: number;
  score: number;
  generated: boolean;
};

type ChartRow = { time: string } & Record<string, number | string>;

type NodeStatus = {
  score: number;
  zScore: number;
  severity: Severity;
  ttf: number;
  lastUpdate: string;
  receivedAt: number;
};

type ServerHealth = {
  status: string;
  model_loaded: boolean;
  llm_enabled: boolean;
  uptime_seconds: number;
  connected_clients: number;
};

const severityStyles: Record<Severity, string> = {
  critical: "bg-red-950/40 border-red-700",
  warning: "bg-amber-950/40 border-amber-700",
  normal: "bg-gray-900 border-gray-800",
};

const severityIconStyles: Record<Severity, string> = {
  critical: "text-red-400",
  warning: "text-amber-400",
  normal: "text-emerald-400",
};

const logStyles: Record<Severity, string> = {
  critical: "bg-red-950/30 border-red-500 text-red-200",
  warning: "bg-amber-950/30 border-amber-500 text-amber-200",
  normal: "bg-gray-800 border-gray-600",
};

export default function MurmurDashboard() {
  const [logs, setLogs] = useState<TelemetryLog[]>([]);
  const [chartData, setChartData] = useState<ChartRow[]>([]);
  const [nodeStatuses, setNodeStatuses] = useState<Record<number, NodeStatus>>(
    {},
  );
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>("connecting");
  const [health, setHealth] = useState<ServerHealth | null>(null);
  const [now, setNow] = useState(() => Date.now());

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const attemptRef = useRef(0);
  // Guards the reconnect loop: the socket's onclose fires *after* React's
  // cleanup runs, so without this an unmount schedules a reconnect that nothing
  // ever cancels — which under StrictMode leaks a socket on every mount.
  const closedByUs = useRef(false);
  const logCounter = useRef(0);

  // Drives the stale-node indicator without re-rendering on every frame.
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 5_000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();

    const checkHealth = async () => {
      try {
        const res = await fetch(`${HEALTH_URL}/health`, {
          signal: controller.signal,
          headers: API_KEY ? { "X-API-Key": API_KEY } : undefined,
        });
        if (!cancelled && res.ok) setHealth(await res.json());
      } catch {
        if (!cancelled) setHealth(null);
      }
    };

    checkHealth();
    const id = setInterval(checkHealth, HEALTH_POLL_MS);
    return () => {
      cancelled = true;
      controller.abort();
      clearInterval(id);
    };
  }, []);

  const handleFrame = useCallback((frame: TelemetryFrame) => {
    const receivedAt = Date.now();
    const time = new Date(receivedAt).toLocaleTimeString();
    const severity: Severity = frame.anomaly?.severity ?? "normal";
    const ttfPercent = (frame.ttf_prediction ?? 0) * 100;
    logCounter.current += 1;

    setLogs((prev) => [
      {
        // A monotonic counter, not Date.now(): several nodes report within the
        // same millisecond, which produced duplicate React keys.
        id: `${receivedAt}-${logCounter.current}`,
        time,
        severity,
        text: frame.telemetry || "",
        nodeId: frame.node_id,
        score: frame.anomaly?.score ?? 0,
        generated: frame.generated ?? false,
      },
      ...prev.slice(0, MAX_LOGS - 1),
    ]);

    setChartData((prev) => {
      const key = `node_${frame.node_id}`;
      const last = prev[prev.length - 1];
      // Nodes report near-simultaneously; merging into the current row keeps
      // one x-position per instant instead of one per message.
      if (last && last.time === time) {
        const merged = { ...last, [key]: Math.round(ttfPercent * 10) / 10 };
        return [...prev.slice(0, -1), merged];
      }
      const row: ChartRow = { time, [key]: Math.round(ttfPercent * 10) / 10 };
      return [...prev.slice(-(MAX_CHART_POINTS - 1)), row];
    });

    setNodeStatuses((prev) => ({
      ...prev,
      [frame.node_id]: {
        score: frame.anomaly?.score ?? 0,
        zScore: frame.anomaly?.z_score ?? 0,
        severity,
        ttf: ttfPercent,
        lastUpdate: time,
        receivedAt,
      },
    }));
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setConnectionStatus("connecting");
    closedByUs.current = false;

    const url = API_KEY
      ? `${WS_URL}?api_key=${encodeURIComponent(API_KEY)}`
      : WS_URL;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      attemptRef.current = 0;
      setConnectionStatus("connected");
    };

    ws.onmessage = (event) => {
      try {
        handleFrame(JSON.parse(event.data) as TelemetryFrame);
      } catch {
        // Ignore malformed frames rather than tearing down the feed.
      }
    };

    ws.onclose = () => {
      setConnectionStatus("disconnected");
      if (closedByUs.current) return;

      // Exponential backoff with jitter. The previous fixed 3s retry hammered a
      // recovering backend from every open dashboard simultaneously.
      const attempt = attemptRef.current++;
      const delay = Math.min(
        RECONNECT_BASE_MS * 2 ** attempt,
        RECONNECT_MAX_MS,
      );
      const jittered = delay * (0.5 + Math.random() * 0.5);
      reconnectTimer.current = setTimeout(connect, jittered);
    };

    ws.onerror = () => ws.close();
  }, [handleFrame]);

  useEffect(() => {
    connect();
    return () => {
      closedByUs.current = true;
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [connect]);

  const activeNodes = useMemo(
    () =>
      Object.keys(nodeStatuses)
        .map(Number)
        .sort((a, b) => a - b),
    [nodeStatuses],
  );

  const statusMeta =
    connectionStatus === "connected"
      ? { dot: "bg-emerald-500", ping: "bg-emerald-400", label: "Cluster Online" }
      : connectionStatus === "connecting"
        ? { dot: "bg-amber-500", ping: "bg-amber-400", label: "Connecting…" }
        : { dot: "bg-red-500", ping: "bg-red-400", label: "Disconnected" };

  return (
    <div className="min-h-screen p-8 font-sans">
      <header className="mb-8 flex flex-wrap items-center justify-between gap-4 border-b border-gray-800 pb-4">
        <div>
          <h1 className="flex items-center gap-3 text-3xl font-bold tracking-tight text-white">
            <Activity className="text-emerald-500" /> Murmur Acoustic Telemetry
          </h1>
          <p className="mt-1 text-gray-400">
            Live Spatio-Temporal Factory Monitoring
          </p>
        </div>

        <div className="flex items-center gap-4">
          {health && (
            <div className="text-right text-xs text-gray-500">
              <div>Uptime {Math.floor(health.uptime_seconds / 60)}m</div>
              <div>
                {health.llm_enabled ? "LLM active" : "Templated telemetry"}
              </div>
            </div>
          )}
          <div className="flex items-center gap-2 rounded-full border border-gray-800 bg-gray-900 px-4 py-2">
            <span className="relative flex h-3 w-3">
              <span
                className={`absolute inline-flex h-full w-full animate-ping rounded-full ${statusMeta.ping} opacity-75`}
              />
              <span
                className={`relative inline-flex h-3 w-3 rounded-full ${statusMeta.dot}`}
              />
            </span>
            <span className="text-sm font-medium">{statusMeta.label}</span>
          </div>
        </div>
      </header>

      {activeNodes.length > 0 && (
        <div className="mb-6 grid grid-cols-2 gap-4 md:grid-cols-4">
          {activeNodes.map((nodeId) => {
            const status = nodeStatuses[nodeId];
            const stale = now - status.receivedAt > STALE_AFTER_MS;
            return (
              <div
                key={nodeId}
                className={`rounded-xl border p-4 shadow-lg transition-opacity ${
                  severityStyles[status.severity]
                } ${stale ? "opacity-40" : ""}`}
              >
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-xs font-medium text-gray-400">
                    Node {nodeId}
                  </span>
                  <ShieldAlert
                    size={14}
                    className={severityIconStyles[status.severity]}
                  />
                </div>
                <div className="text-2xl font-bold">
                  {status.ttf.toFixed(1)}%
                </div>
                <div className="mt-1 font-mono text-xs text-gray-500">
                  score {status.score.toFixed(4)} · z {status.zScore.toFixed(2)}
                </div>
                <div className="mt-0.5 text-xs text-gray-600">
                  {stale ? "stale — no recent data" : status.lastUpdate}
                </div>
              </div>
            );
          })}
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="rounded-xl border border-gray-800 bg-gray-900 p-6 shadow-xl lg:col-span-2">
          <h2 className="mb-4 flex items-center gap-2 text-lg font-semibold">
            <AlertTriangle className="text-amber-500" size={20} />
            Liquid Network Failure Forecast (TTF)
          </h2>
          <div className="h-80 w-full">
            {chartData.length === 0 ? (
              <div className="flex h-full items-center justify-center text-gray-500">
                <WifiOff className="mr-2" size={16} />
                {connectionStatus === "disconnected"
                  ? "Backend offline — no live data."
                  : "Waiting for model predictions…"}
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="time" stroke="#9CA3AF" fontSize={12} />
                  <YAxis
                    stroke="#9CA3AF"
                    fontSize={12}
                    domain={[0, 100]}
                    unit="%"
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#1F2937",
                      border: "none",
                      borderRadius: "8px",
                    }}
                  />
                  <Legend />
                  {activeNodes.map((nodeId) => (
                    <Line
                      key={nodeId}
                      type="monotone"
                      dataKey={`node_${nodeId}`}
                      name={`Node ${nodeId}`}
                      stroke={NODE_COLORS[nodeId % NODE_COLORS.length]}
                      strokeWidth={2}
                      dot={false}
                      connectNulls
                      activeDot={{ r: 6 }}
                      isAnimationActive={false}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        <div className="flex flex-col overflow-hidden rounded-xl border border-gray-800 bg-gray-900 p-6 shadow-xl">
          <h2 className="mb-4 flex items-center gap-2 text-lg font-semibold">
            <CheckCircle className="text-blue-500" size={20} />
            Audio LLM Diagnostics
          </h2>
          <div className="scroll-slim max-h-[360px] flex-1 space-y-3 overflow-y-auto pr-2">
            {logs.length === 0 ? (
              <p className="text-sm text-gray-500">
                Waiting for incoming telemetry…
              </p>
            ) : (
              logs.map((log) => (
                <div
                  key={log.id}
                  className={`rounded-lg border-l-4 p-3 text-sm ${logStyles[log.severity]}`}
                >
                  <div className="mb-1 flex justify-between text-xs opacity-70">
                    <span>Node {log.nodeId}</span>
                    <span className="flex items-center gap-2">
                      {!log.generated && (
                        <span
                          className="rounded bg-gray-700 px-1 text-[10px] uppercase tracking-wide"
                          title="Templated — no LLM resident"
                        >
                          template
                        </span>
                      )}
                      {log.time}
                    </span>
                  </div>
                  <p>{log.text}</p>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
