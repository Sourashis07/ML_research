import React from 'react';
import { ShieldCheck, AlertTriangle, Sun, Moon, Compass, Radio } from 'lucide-react';

export default function HUDHeader({ isAnomaly, orbitState }) {
  const lat = orbitState?.lat?.toFixed(2) || '0.00';
  const lon = orbitState?.lon?.toFixed(2) || '0.00';
  const statusStr = orbitState?.eclipse_status || 'SUNLIT (DAYLIGHT)';
  const isSunlit = statusStr.includes('SUNLIT');
  const solarFlux = orbitState?.solar_flux?.toFixed(1) || '1361.0';

  return (
    <div className="glass-card" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px', alignItems: 'center' }}>
      <div>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px' }}>SPACECRAFT STATUS</div>
        <div style={{ marginTop: '6px' }}>
          {isAnomaly ? (
            <div className="badge-critical title-orbitron">
              <AlertTriangle size={18} /> CRITICAL ANOMALY DETECTED
            </div>
          ) : (
            <div className="badge-nominal title-orbitron">
              <ShieldCheck size={18} /> SPACECRAFT NOMINAL
            </div>
          )}
        </div>
      </div>

      <div>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Compass size={14} color="var(--accent-cyan)" /> ORBITAL COORDINATES
        </div>
        <div className="mono-font" style={{ fontSize: '1.4rem', fontWeight: 700, marginTop: '4px' }}>
          {lat}° N, {lon}° E
        </div>
      </div>

      <div>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px', display: 'flex', alignItems: 'center', gap: '6px' }}>
          {isSunlit ? <Sun size={14} color="var(--accent-gold)" /> : <Moon size={14} color="var(--accent-cyan)" />} ECLIPSE ZONE
        </div>
        <div className="mono-font" style={{ fontSize: '1.2rem', fontWeight: 700, marginTop: '4px', color: isSunlit ? 'var(--accent-gold)' : 'var(--accent-cyan)' }}>
          {statusStr}
        </div>
      </div>

      <div>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Radio size={14} color="var(--accent-green)" /> INSTANTANEOUS SOLAR FLUX
        </div>
        <div className="mono-font" style={{ fontSize: '1.4rem', fontWeight: 700, marginTop: '4px', color: '#fff' }}>
          {solarFlux} <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>W/m²</span>
        </div>
      </div>
    </div>
  );
}
