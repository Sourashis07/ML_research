import React from 'react';
import { Play, Pause, RotateCcw, Zap, AlertCircle } from 'lucide-react';

export default function SimulatorControls({ 
  isPlaying, onTogglePlay, onReset, 
  timeWarp, onChangeTimeWarp, 
  faultType, onChangeFault 
}) {
  const FAULT_OPTIONS = [
    { value: 'NONE', label: '✓ NOMINAL (NO FAULT)' },
    { value: 'SOLAR_FLARE_CME', label: '☀️ SOLAR FLARE / CME EVENT' },
    { value: 'BATTERY_CELL_DROP', label: '🔋 BATTERY CELL DROP' },
    { value: 'SOLAR_ARRAY_SHUNT', label: '⚡ SOLAR ARRAY SHUNT' },
    { value: 'RADIATOR_VALVE_STICK', label: '🌡️ RADIATOR VALVE STICK' },
    { value: 'THERMISTOR_DRIFT', label: '📈 THERMISTOR DRIFT' }
  ];

  return (
    <div className="glass-card" style={{ height: '420px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
      <div>
        <div className="title-orbitron" style={{ fontSize: '1rem', color: 'var(--accent-cyan)', marginBottom: '16px' }}>
          🎛️ Simulator & Space Weather Controls
        </div>

        {/* Orbit Controls */}
        <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
          <button className="btn-cyan" style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }} onClick={onTogglePlay}>
            {isPlaying ? <><Pause size={16} /> PAUSE</> : <><Play size={16} /> PLAY ORBIT</>}
          </button>
          <button className="btn-dark" onClick={onReset} title="Reset Orbit & Faults">
            <RotateCcw size={16} />
          </button>
        </div>

        {/* Time Warp Slider */}
        <div style={{ marginBottom: '24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '6px' }}>
            <span>Time Warp Factor:</span>
            <span className="mono-font" style={{ color: 'var(--accent-cyan)' }}>{timeWarp}x</span>
          </div>
          <input 
            type="range" min="1" max="25" step="4" 
            value={timeWarp} 
            onChange={(e) => onChangeTimeWarp(Number(e.target.value))}
            style={{ width: '100%', accentColor: 'var(--accent-cyan)', cursor: 'pointer' }}
          />
        </div>

        <hr style={{ borderColor: 'rgba(255,255,255,0.1)', marginBottom: '20px' }} />

        {/* Fault Injection Selector */}
        <div>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Zap size={15} color="var(--accent-gold)" /> INJECT SPACE WEATHER / HARDWARE FAULT:
          </div>
          <select 
            value={faultType} 
            onChange={(e) => onChangeFault(e.target.value)}
            style={{
              width: '100%',
              backgroundColor: 'rgba(5, 10, 20, 0.9)',
              color: '#fff',
              border: '1px solid var(--panel-border)',
              padding: '10px 12px',
              borderRadius: '6px',
              fontFamily: 'Rajdhani, sans-serif',
              fontSize: '0.95rem',
              outline: 'none',
              cursor: 'pointer'
            }}
          >
            {FAULT_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        </div>
      </div>

      <div style={{ background: faultType !== 'NONE' ? 'rgba(255, 42, 95, 0.15)' : 'rgba(0, 255, 102, 0.08)', border: `1px solid ${faultType !== 'NONE' ? 'var(--accent-red)' : 'var(--accent-green)'}`, padding: '12px', borderRadius: '8px', fontSize: '0.85rem' }}>
        <div style={{ fontWeight: 700, color: faultType !== 'NONE' ? 'var(--accent-red)' : 'var(--accent-green)', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <AlertCircle size={15} /> Active Fault Injection Mode:
        </div>
        <div className="mono-font" style={{ marginTop: '4px', color: '#fff' }}>
          {faultType}
        </div>
      </div>
    </div>
  );
}
