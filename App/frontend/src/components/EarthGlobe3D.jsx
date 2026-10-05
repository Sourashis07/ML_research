import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';

export default function EarthGlobe3D({ orbitState, orbitPath, isAnomaly }) {
  const mountRef = useRef(null);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth || 500;
    const height = container.clientHeight || 400;

    // Scene, Camera, Renderer
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 10000);
    camera.position.set(0, 0, 18);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    // Ambient & Directional Lights
    const ambientLight = new THREE.AmbientLight(0x1a2b4c, 1.8);
    scene.add(ambientLight);

    const sunLight = new THREE.DirectionalLight(0xffffff, 2.5);
    sunLight.position.set(20, 10, 20);
    scene.add(sunLight);

    // Earth Sphere
    const r_earth = 5;
    const geometry = new THREE.SphereGeometry(r_earth, 64, 64);
    const material = new THREE.MeshPhongMaterial({
      color: 0x0b3954,
      emissive: 0x04152d,
      specular: 0x00a8e8,
      shininess: 25,
      wireframe: false
    });
    const earthMesh = new THREE.Mesh(geometry, material);
    scene.add(earthMesh);

    // Earth Atmosphere Glow Ring
    const atmoGeo = new THREE.SphereGeometry(r_earth * 1.05, 32, 32);
    const atmoMat = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      transparent: true,
      opacity: 0.12,
      side: THREE.BackSide
    });
    const atmoMesh = new THREE.Mesh(atmoGeo, atmoMat);
    scene.add(atmoMesh);

    // Orbit Path Line
    let orbitLineMesh = null;
    if (orbitPath && orbitPath.x_eci) {
      const points = [];
      const scale = 5.0 / 6371.0;
      for (let i = 0; i < orbitPath.x_eci.length; i++) {
        points.push(new THREE.Vector3(
          orbitPath.x_eci[i] * scale,
          orbitPath.z_eci[i] * scale,
          -orbitPath.y_eci[i] * scale
        ));
      }
      const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
      const lineMat = new THREE.LineBasicMaterial({ color: 0x00f0ff, linewidth: 2 });
      orbitLineMesh = new THREE.LineLoop(lineGeo, lineMat);
      scene.add(orbitLineMesh);
    }

    // Satellite Position Marker
    const satGeo = new THREE.SphereGeometry(0.25, 16, 16);
    const satMat = new THREE.MeshBasicMaterial({
      color: isAnomaly ? 0xff2a5f : 0x00ff66
    });
    const satMesh = new THREE.Mesh(satGeo, satMat);
    scene.add(satMesh);

    // Position Satellite
    if (orbitState && orbitState.x_eci !== undefined) {
      const scale = 5.0 / 6371.0;
      satMesh.position.set(
        orbitState.x_eci * scale,
        orbitState.z_eci * scale,
        -orbitState.y_eci * scale
      );
    }

    // Mouse Controls / Rotation Logic
    let isDragging = false;
    let previousMousePosition = { x: 0, y: 0 };

    const handleMouseDown = (e) => {
      isDragging = true;
      previousMousePosition = { x: e.clientX, y: e.clientY };
    };

    const handleMouseMove = (e) => {
      if (!isDragging) return;
      const deltaMove = {
        x: e.clientX - previousMousePosition.x,
        y: e.clientY - previousMousePosition.y
      };

      earthMesh.rotation.y += deltaMove.x * 0.005;
      earthMesh.rotation.x += deltaMove.y * 0.005;
      if (orbitLineMesh) {
        orbitLineMesh.rotation.y += deltaMove.x * 0.005;
        orbitLineMesh.rotation.x += deltaMove.y * 0.005;
      }
      previousMousePosition = { x: e.clientX, y: e.clientY };
    };

    const handleMouseUp = () => { isDragging = false; };

    const domElem = renderer.domElement;
    domElem.addEventListener('mousedown', handleMouseDown);
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    // Animation Loop
    let animId;
    const animate = () => {
      animId = requestAnimationFrame(animate);
      earthMesh.rotation.y += 0.001; // Slow continuous rotation
      renderer.render(scene, camera);
    };
    animate();

    // Cleanup
    return () => {
      cancelAnimationFrame(animId);
      domElem.removeEventListener('mousedown', handleMouseDown);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
      geometry.dispose();
      material.dispose();
      renderer.dispose();
    };
  }, [orbitState, orbitPath, isAnomaly]);

  return (
    <div className="glass-card" style={{ height: '420px', position: 'relative', overflow: 'hidden' }}>
      <div className="title-orbitron" style={{ fontSize: '1rem', color: 'var(--accent-cyan)', marginBottom: '8px', display: 'flex', justifyContent: 'space-between' }}>
        <span>🌍 3D Orbital Digital Twin & Satellite Propagation</span>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Drag to rotate 360°</span>
      </div>
      <div ref={mountRef} style={{ width: '100%', height: 'calc(100% - 30px)', cursor: 'grab' }} />
    </div>
  );
}
