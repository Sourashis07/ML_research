import React, { useState } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, 
  BarChart, Bar, Cell 
} from 'recharts';

export default function TelemetryCharts({ telemetryData }) {
  const [scrubIndex, setScrubIndex] = useState(0);

  if (!telemetryData || !telemetryData.actual_col0) {
    return <div className="glass-card" style={{ padding: '40px', textAlign: 'center' }}>Loading Telemetry Pipeline...</div>;
  }

  const { actual_col0, recon_col0, residuals, threshold, attributions, subsystem_sparklines } = telemetryData;

  // Format data for Chart 1 & Chart 2
  const lineChartData = actual_col0.map((val, idx) => ({
    step: idx,
    actual: Number(val.toFixed(4)),
    recon: Number(recon_col0[idx]?.toFixed(4) || 0),
    residual: Number(residuals[idx]?.toFixed(4) || 0),
    threshold: Number(threshold.toFixed(4)),
    ch0: subsystem_sparklines?.channel_0?.[idx] || 0,
    ch1: subsystem_sparklines?.channel_1?.[idx] || 0,
    ch2: subsystem_sparklines?.channel_2?.[idx] || 0,
  }));

  const maxIndex = lineChartData.length - 1;
  const currentScrub = Math.min(scrubIndex, maxIndex);

  // Top Attributions data for Chart 3
  const topAttributions = (attributions || []).slice(0, 5).map(item => ({
    name: item.channel_name,
    score: Number(item.contribution_pct.toFixed(1))
  }));

  return (
    <div>
      {/* Time Scrubbing Controller */}
      <div className="glass-card" style={{ marginBottom: '20px', padding: '14px 20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem', marginBottom: '6px' }}>
          <span className="title-orbitron" style={{ color: 'var(--accent-cyan)' }}>⏱️ Synchronized Time Scrubbing Cursor</span>
          <span className="mono-font">Step: {currentScrub} / {maxIndex}</span>
        </div>
        <input 
          type="range" min="0" max={maxIndex} 
          value={currentScrub} 
          onChange={(e) => setScrubIndex(Number(e.target.value))}
          style={{ width: '100%', accentColor: 'var(--accent-cyan)', cursor: 'pointer' }}
        />
      </div>

      {/* 4-Chart Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))', gap: '20px' }}>
        
        {/* Chart 1: Expectation vs Reality Overlay */}
        <div className="glass-card" style={{ height: '300px' }}>
          <div style={{ fontSize: '0.85rem', color: 'var(--accent-cyan)', fontWeight: 700, marginBottom: '8px' }}>
            CHART 1: Telemetry Overlay (Actual vs. AI Reconstruction)
          </div>
          <ResponsiveContainer width="100%" height="85%">
            <LineChart data={lineChartData} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
              <XAxis dataKey="step" stroke="#94a3b8" fontSize={11} />
              <YAxis stroke="#94a3b8" fontSize={11} domain={['auto', 'auto']} />
              <Tooltip contentStyle={{ backgroundColor: '#050a14', borderColor: '#00f0ff', fontSize: '12px' }} />
              <Line type="monotone" dataKey="actual" stroke="#00f0ff" strokeWidth={1.5} dot={false} name="Actual Telemetry" />
              <Line type="monotone" dataKey="recon" stroke="#ffb700" strokeWidth={1.5} strokeDasharray="3 3" dot={false} name="AI Reconstruction" />
              <ReferenceLine x={currentScrub} stroke="#ffffff" strokeDasharray="3 3" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Chart 2: Residual Error Profile */}
        <div className="glass-card" style={{ height: '300px' }}>
          <div style={{ fontSize: '0.85rem', color: 'var(--accent-red)', fontWeight: 700, marginBottom: '8px' }}>
            CHART 2: Residual Error Profile e_t & Dynamic Bounds (μ+3σ)
          </div>
          <ResponsiveContainer width="100%" height="85%">
            <LineChart data={lineChartData} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
              <XAxis dataKey="step" stroke="#94a3b8" fontSize={11} />
              <YAxis stroke="#94a3b8" fontSize={11} domain={['auto', 'auto']} />
              <Tooltip contentStyle={{ backgroundColor: '#050a14', borderColor: '#ff2a5f', fontSize: '12px' }} />
              <Line type="monotone" dataKey="residual" stroke="#ff2a5f" strokeWidth={1.5} dot={false} name="Smoothed Error" />
              <Line type="monotone" dataKey="threshold" stroke="#00ff66" strokeWidth={1.5} strokeDasharray="4 4" dot={false} name="Dynamic Threshold" />
              <ReferenceLine x={currentScrub} stroke="#ffffff" strokeDasharray="3 3" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Chart 3: Root-Cause Attribution Bar Chart */}
        <div className="glass-card" style={{ height: '300px' }}>
          <div style={{ fontSize: '0.85rem', color: 'var(--accent-gold)', fontWeight: 700, marginBottom: '8px' }}>
            CHART 3: Top Failing Subsystem Root-Cause Attribution Ranking
          </div>
          <ResponsiveContainer width="100%" height="85%">
            <BarChart data={topAttributions} layout="vertical" margin={{ top: 5, right: 20, left: 40, bottom: 5 }}>
              <XAxis type="number" stroke="#94a3b8" fontSize={11} unit="%" />
              <YAxis type="category" dataKey="name" stroke="#94a3b8" fontSize={11} width={120} />
              <Tooltip contentStyle={{ backgroundColor: '#050a14', borderColor: '#ffb700', fontSize: '12px' }} />
              <Bar dataKey="score" fill="#ff2a5f" radius={[0, 4, 4, 0]}>
                {topAttributions.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={index === 0 ? '#ff2a5f' : '#00f0ff'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Chart 4: Multi-Sensor Subsystem Grid */}
        <div className="glass-card" style={{ height: '300px' }}>
          <div style={{ fontSize: '0.85rem', color: 'var(--accent-green)', fontWeight: 700, marginBottom: '8px' }}>
            CHART 4: Multi-Sensor Subsystem Grid Sparklines (Power / Thermal / Cmd)
          </div>
          <ResponsiveContainer width="100%" height="85%">
            <LineChart data={lineChartData} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
              <XAxis dataKey="step" stroke="#94a3b8" fontSize={11} />
              <YAxis stroke="#94a3b8" fontSize={11} />
              <Tooltip contentStyle={{ backgroundColor: '#050a14', borderColor: '#00ff66', fontSize: '12px' }} />
              <Line type="monotone" dataKey="ch0" stroke="#00f0ff" strokeWidth={1} dot={false} name="Power (V_bus)" />
              <Line type="monotone" dataKey="ch1" stroke="#ffb700" strokeWidth={1} dot={false} name="Thermal (T_batt)" />
              <Line type="monotone" dataKey="ch2" stroke="#00ff66" strokeWidth={1} dot={false} name="Avionics (Cmd)" />
              <ReferenceLine x={currentScrub} stroke="#ffffff" strokeDasharray="3 3" />
            </LineChart>
          </ResponsiveContainer>
        </div>

      </div>
    </div>
  );
}
