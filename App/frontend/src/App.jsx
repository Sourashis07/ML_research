import React, { useState, useEffect } from 'react';
import axios from 'axios';
import HUDHeader from './components/HUDHeader';
import EarthGlobe3D from './components/EarthGlobe3D';
import SimulatorControls from './components/SimulatorControls';
import TelemetryCharts from './components/TelemetryCharts';
import BenchmarkView from './components/BenchmarkView';
import { Satellite, Activity, BarChart2 } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('mission');
  const [channels, setChannels] = useState([]);
  const [selectedChannel, setSelectedChannel] = useState('P-1');
  const [metadata, setMetadata] = useState([]);
  const [telemetryData, setTelemetryData] = useState(null);
  
  // Simulator State
  const [simTime, setSimTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [timeWarp, setTimeWarp] = useState(1);
  const [faultType, setFaultType] = useState('NONE');
  const [orbitState, setOrbitState] = useState(null);
  const [orbitPath, setOrbitPath] = useState(null);

  // Fetch Channels on Load
  useEffect(() => {
    axios.get('/api/channels')
      .then(res => {
        setChannels(res.data.channels || ['P-1']);
        setMetadata(res.data.metadata || []);
      })
      .catch(err => console.error(err));
  }, []);

  // Fetch Telemetry Inference when Channel or Fault changes
  useEffect(() => {
    axios.get(`/api/telemetry/${selectedChannel}?fault_type=${faultType}`)
      .then(res => setTelemetryData(res.data))
      .catch(err => console.error(err));
  }, [selectedChannel, faultType]);

  // Orbit Propagation Animation Loop
  useEffect(() => {
    let interval = null;
    if (isPlaying) {
      interval = setInterval(() => {
        setSimTime(prev => prev + 15 * timeWarp);
      }, 300);
    }
    return () => clearInterval(interval);
  }, [isPlaying, timeWarp]);

  // Propagate Orbit state when simTime changes
  useEffect(() => {
    axios.post('/api/simulator/propagate', { elapsed_seconds: simTime })
      .then(res => {
        setOrbitState(res.data.current_state);
        setOrbitPath(res.data.orbit_path);
      })
      .catch(err => console.error(err));
  }, [simTime]);

  const handleReset = () => {
    setSimTime(0);
    setIsPlaying(false);
    setFaultType('NONE');
  };

  return (
    <div style={{ maxWidth: '1440px', margin: '0 auto', padding: '24px' }}>
      
      {/* Top Navbar */}
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <Satellite size={32} color="var(--accent-cyan)" />
          <div>
            <h1 className="title-orbitron" style={{ fontSize: '1.6rem', background: 'linear-gradient(135deg, #00f0ff 0%, #7000ff 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
              AUTONOMOUS SPACECRAFT MISSION CONTROL
            </h1>
            <div style={{ fontSize: '0.8rem', color: 'var(--accent-cyan)', letterSpacing: '1px' }}>
              SPATIAL-TEMPORAL ATTENTION AUTOENCODER & 3D DIGITAL TWIN
            </div>
          </div>
        </div>

        {/* View Switcher Tabs */}
        <div style={{ display: 'flex', gap: '8px', background: 'rgba(13, 25, 48, 0.8)', padding: '6px', borderRadius: '8px', border: '1px solid var(--panel-border)' }}>
          <button 
            className={`btn-dark ${activeTab === 'mission' ? 'btn-cyan' : ''}`}
            onClick={() => setActiveTab('mission')}
            style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
          >
            <Activity size={16} /> Live Mission Control
          </button>
          <button 
            className={`btn-dark ${activeTab === 'benchmark' ? 'btn-cyan' : ''}`}
            onClick={() => setActiveTab('benchmark')}
            style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
          >
            <BarChart2 size={16} /> Benchmark & Analytics
          </button>
        </div>
      </header>

      {/* Main Content Views */}
      {activeTab === 'mission' ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          
          {/* Channel Selector Bar */}
          <div className="glass-card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px 20px' }}>
            <div style={{ fontWeight: 700, color: 'var(--text-muted)' }}>SELECT TELEMETRY CHANNEL:</div>
            <select 
              value={selectedChannel} 
              onChange={(e) => setSelectedChannel(e.target.value)}
              style={{
                backgroundColor: '#050a14', color: 'var(--accent-cyan)', border: '1px solid var(--panel-border)',
                padding: '6px 16px', borderRadius: '6px', fontFamily: 'Share Tech Mono, monospace', fontSize: '1rem'
              }}
            >
              {channels.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>

          {/* Top HUD Header */}
          <HUDHeader isAnomaly={telemetryData?.is_anomaly} orbitState={orbitState} />

          {/* 3D Earth Globe & Simulator Controls Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: '24px' }}>
            <EarthGlobe3D orbitState={orbitState} orbitPath={orbitPath} isAnomaly={telemetryData?.is_anomaly} />
            <SimulatorControls 
              isPlaying={isPlaying} 
              onTogglePlay={() => setIsPlaying(!isPlaying)} 
              onReset={handleReset}
              timeWarp={timeWarp}
              onChangeTimeWarp={setTimeWarp}
              faultType={faultType}
              onChangeFault={setFaultType}
            />
          </div>

          {/* Synchronized Telemetry Dashboard */}
          <TelemetryCharts telemetryData={telemetryData} />

        </div>
      ) : (
        <BenchmarkView metadata={metadata} />
      )}

    </div>
  );
}
