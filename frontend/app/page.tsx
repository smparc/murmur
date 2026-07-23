"use client";

import { useState, useEffect, useRef, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Activity, AlertTriangle, CheckCircle, WifiOff, ShieldAlert } from 'lucide-react';


const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws/telemetry';
const HEALTH_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';


type TelemetryLog = {
  id: number;
  time: string;
  level: 'CRITICAL' | 'WARNING' | 'INFO';
  text: string;
  node_id: number;
  anomaly_score: number;
};


type TTFDataPoint = {
  time: string;
  probability: number;
  node_id: number;
};


type ConnectionStatus = 'connected' | 'connecting' | 'disconnected';


type NodeStatus = {
  anomaly_score: number;
  severity: string;
  ttf: number;
  last_update: string;
};


export default function MurmurDashboard() {
  const [telemetryLogs, setTelemetryLogs] = useState<TelemetryLog[]>([]);
  const [ttfData, setTtfData] = useState<TTFDataPoint[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('connecting');
  const [serverHealth, setServerHealth] = useState<{ model_loaded: boolean; uptime_seconds: number } | null>(null);
  const [nodeStatuses, setNodeStatuses] = useState<Record<number, NodeStatus>>({});
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeout = useRef<NodeJS.Timeout>();


  // Health check polling
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${HEALTH_URL}/health`);
        if (res.ok) {
          const data = await res.json();
          setServerHealth(data);
        }
      } catch {
        setServerHealth(null);
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 10_000);
    return () => clearInterval(interval);
  }, []);


  // WebSocket connection with auto-reconnect
  const connectWs = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;


    setConnectionStatus('connecting');
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;


    ws.onopen = () => {
      setConnectionStatus('connected');
    };


    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        const now = new Date().toLocaleTimeString();


        // Use structured anomaly data from the backend (not regex heuristics)
        const anomaly = data.anomaly || {};
        const severity: string = anomaly.severity || 'normal';
        const level: TelemetryLog['level'] = severity === 'critical'
          ? 'CRITICAL'
          : severity === 'warning'
            ? 'WARNING'
            : 'INFO';


        // TTF comes directly from the LNN model prediction, not text analysis
        const ttfProbability: number = (data.ttf_prediction ?? 0) * 100;


        // Add to telemetry log feed
        setTelemetryLogs(prev => [
          {
            id: Date.now(),
            time: now,
            level,
            text: data.telemetry || '',
            node_id: data.node_id,
            anomaly_score: anomaly.score ?? 0,
          },
          ...prev.slice(0, 49),
        ]);


        // TTF chart data — real model predictions
        setTtfData(prev => [
          ...prev.slice(-59),
          { time: now, probability: Math.round(ttfProbability * 10) / 10, node_id: data.node_id },
        ]);


        // Update per-node status
        setNodeStatuses(prev => ({
          ...prev,
          [data.node_id]: {
            anomaly_score: anomaly.score ?? 0,
            severity,
            ttf: ttfProbability,
            last_update: now,
          },
        }));
      } catch {
        // ignore malformed frames
      }
    };


    ws.onclose = () => {
      setConnectionStatus('disconnected');
      // Auto-reconnect after 3 seconds
      reconnectTimeout.current = setTimeout(connectWs, 3000);
    };


    ws.onerror = () => {
      ws.close();
    };
  }, []);

  useEffect(() => {
    connectWs();
    return () => {
      clearTimeout(reconnectTimeout.current);
      wsRef.current?.close();
    };
  }, [connectWs]);


  // NO fake data fallback — safety-critical system must only show real data
  // When disconnected, we display the "Waiting for data" placeholder instead


  const statusColor = connectionStatus === 'connected' ? 'bg-emerald-500' : connectionStatus === 'connecting' ? 'bg-amber-500' : 'bg-red-500';
  const statusPing = connectionStatus === 'connected' ? 'bg-emerald-400' : connectionStatus === 'connecting' ? 'bg-amber-400' : 'bg-red-400';
  const statusLabel = connectionStatus === 'connected' ? 'Cluster Online' : connectionStatus === 'connecting' ? 'Connecting…' : 'Disconnected';

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-8 font-sans">
      <header className="mb-8 flex items-center justify-between border-b border-gray-800 pb-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
            <Activity className="text-emerald-500" /> Murmur Acoustic Telemetry
          </h1>
          <p className="text-gray-400 mt-1">Live Spatio-Temporal Factory Monitoring</p>
        </div>
        <div className="flex items-center gap-4">
          {serverHealth && (
            <span className="text-xs text-gray-500">
              Uptime: {Math.floor(serverHealth.uptime_seconds / 60)}m
            </span>
          )}
          <div className="flex items-center gap-2 bg-gray-900 px-4 py-2 rounded-full border border-gray-800">
            <span className="relative flex h-3 w-3">
              <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${statusPing} opacity-75`}></span>
              <span className={`relative inline-flex rounded-full h-3 w-3 ${statusColor}`}></span>
            </span>
            <span className="text-sm font-medium">{statusLabel}</span>
          </div>
        </div>
      </header>

      {/* Node Status Cards */}
      {Object.keys(nodeStatuses).length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          {Object.entries(nodeStatuses).map(([nodeId, status]) => (
            <div key={nodeId} className={`p-4 rounded-xl border shadow-lg ${
              status.severity === 'critical' ? 'bg-red-950/40 border-red-700'
              : status.severity === 'warning' ? 'bg-amber-950/40 border-amber-700'
              : 'bg-gray-900 border-gray-800'
            }`}>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-gray-400 font-medium">Node {nodeId}</span>
                <ShieldAlert size={14} className={
                  status.severity === 'critical' ? 'text-red-400'
                  : status.severity === 'warning' ? 'text-amber-400'
                  : 'text-emerald-400'
                } />
              </div>
              <div className="text-2xl font-bold">
                {status.ttf.toFixed(1)}%
              </div>
              <div className="text-xs text-gray-500 mt-1">
                Score: {status.anomaly_score.toFixed(4)} | {status.last_update}
              </div>
            </div>
          ))}
        </div>
      )}


      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* LNN Forecasting Chart */}
        <div className="lg:col-span-2 bg-gray-900 p-6 rounded-xl border border-gray-800 shadow-xl">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <AlertTriangle className="text-amber-500" size={20} />
            Liquid Network Failure Forecast (TTF)
          </h2>
          <div className="h-80 w-full">
            {ttfData.length === 0 ? (
              <div className="h-full flex items-center justify-center text-gray-500">
                <WifiOff className="mr-2" size={16} />
                {connectionStatus === 'disconnected'
                  ? 'Backend offline — no live data. Connect to see real predictions.'
                  : 'Waiting for model predictions…'}
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={ttfData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="time" stroke="#9CA3AF" fontSize={12} />
                  <YAxis stroke="#9CA3AF" fontSize={12} domain={[0, 100]} unit="%" />
                  <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }} />
                  <Line type="monotone" dataKey="probability" stroke="#10B981" strokeWidth={3} dot={false} activeDot={{ r: 8 }} />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        {/* LLM Autoregressive Text Logs */}
        <div className="bg-gray-900 p-6 rounded-xl border border-gray-800 shadow-xl overflow-hidden flex flex-col">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <CheckCircle className="text-blue-500" size={20} />
            Audio LLM Diagnostics
          </h2>
          <div className="flex-1 overflow-y-auto pr-2 space-y-3 max-h-[360px]">
            {telemetryLogs.length === 0 ? (
              <p className="text-gray-500 text-sm">Waiting for incoming telemetry…</p>
            ) : (
              telemetryLogs.map((log) => (
                <div key={log.id} className={`p-3 rounded-lg text-sm border-l-4 ${
                  log.level === 'CRITICAL' ? 'bg-red-950/30 border-red-500 text-red-200'
                  : log.level === 'WARNING' ? 'bg-amber-950/30 border-amber-500 text-amber-200'
                  : 'bg-gray-800 border-gray-600'
                }`}>
                  <div className="flex justify-between text-xs mb-1 opacity-70">
                    <span>Node {log.node_id}</span>
                    <span>{log.time}</span>
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