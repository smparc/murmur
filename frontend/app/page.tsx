"use client";

import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Activity, AlertTriangle, CheckCircle } from 'lucide-react';

export default function MurmurDashboard() {
  const [telemetryLogs, setTelemetryLogs] = useState([]);
  const [ttfData, setTtfData] = useState([]);

  // Simulate WebSocket connection to the Kubernetes inference endpoint
  useEffect(() => {
    const interval = setInterval(() => {
      const now = new Date().toLocaleTimeString();
      const mockProbability = Math.random() * 100;
      
      setTtfData(prev => [...prev.slice(-19), { time: now, probability: mockProbability }]);
      
      if (mockProbability > 85) {
        setTelemetryLogs(prev => [
          { id: Date.now(), time: now, level: 'CRITICAL', text: 'High-frequency oscillation detected in HVAC Unit 4. Cavitation imminent.' },
          ...prev.slice(0, 9)
        ]);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-8 font-sans">
      <header className="mb-8 flex items-center justify-between border-b border-gray-800 pb-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
            <Activity className="text-emerald-500" /> Murmur Acoustic Telemetry
          </h1>
          <p className="text-gray-400 mt-1">Live Spatio-Temporal Factory Monitoring</p>
        </div>
        <div className="flex items-center gap-2 bg-gray-900 px-4 py-2 rounded-full border border-gray-800">
          <span className="relative flex h-3 w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
          </span>
          <span className="text-sm font-medium">Cluster Online</span>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* LNN Forecasting Chart */}
        <div className="lg:col-span-2 bg-gray-900 p-6 rounded-xl border border-gray-800 shadow-xl">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <AlertTriangle className="text-amber-500" size={20} />
            Liquid Network Failure Forecast (TTF)
          </h2>
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={ttfData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="time" stroke="#9CA3AF" fontSize={12} />
                <YAxis stroke="#9CA3AF" fontSize={12} domain={[0, 100]} />
                <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }} />
                <Line type="monotone" dataKey="probability" stroke="#10B981" strokeWidth={3} dot={false} activeDot={{ r: 8 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* LLM Autoregressive Text Logs */}
        <div className="bg-gray-900 p-6 rounded-xl border border-gray-800 shadow-xl overflow-hidden flex flex-col">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <CheckCircle className="text-blue-500" size={20} />
            Audio LLM Diagnostics
          </h2>
          <div className="flex-1 overflow-y-auto pr-2 space-y-3">
            {telemetryLogs.length === 0 ? (
              <p className="text-gray-500 text-sm">Waiting for incoming telemetry...</p>
            ) : (
              telemetryLogs.map((log) => (
                <div key={log.id} className={`p-3 rounded-lg text-sm border-l-4 ${log.level === 'CRITICAL' ? 'bg-red-950/30 border-red-500 text-red-200' : 'bg-gray-800 border-gray-600'}`}>
                  <div className="flex justify-between text-xs mb-1 opacity-70">
                    <span>Node 2</span>
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