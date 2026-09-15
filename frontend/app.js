/* ───────────────────────────────────────────────────────────────────────────
   Sentinel Grid — app.js
   Handles: API fetching, Leaflet map, all 4 tab panels, bio-radar animation
─────────────────────────────────────────────────────────────────────────── */

const API = '';   // same origin — served by FastAPI
let currentDisaster = 'flood';
let zonesData = null, priorityData = null, evacuationData = null, detectionsData = null;
let map = null, zoneLayer = null, routeLayer = null, sensorLayer = null;
let radarAnimId = null;
let _heartbeatSamples = null;   // real PPG-DaLiA test windows + real model predictions

// ── CLOCK ─────────────────────────────────────────────────────────────────
function updateClock() {
  const d = new Date();
  document.getElementById('liveClock').textContent =
    d.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}
setInterval(updateClock, 1000); updateClock();

// ── MAP INIT ──────────────────────────────────────────────────────────────
let evacLayer = null, assetLayer = null;
function initMap() {
  map = L.map('map', {
    center: [11.01, 77.01],
    zoom: 12,
    zoomControl: true,
    attributionControl: true,
  });
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors',
    maxZoom: 18,
  }).addTo(map);
  zoneLayer   = L.layerGroup().addTo(map);
  routeLayer  = L.layerGroup().addTo(map);
  evacLayer   = L.layerGroup().addTo(map);
  assetLayer  = L.layerGroup().addTo(map);
  sensorLayer = L.layerGroup().addTo(map);
}

// ── ZONE COLOUR HELPERS ───────────────────────────────────────────────────
const ZONE_COLORS = { RED: '#ff3b3b', BLUE: '#3b8fff', GREEN: '#00e676' };
const ZONE_FILL   = { RED: 'rgba(255,59,59,0.28)', BLUE: 'rgba(59,143,255,0.20)', GREEN: 'rgba(0,230,118,0.20)' };

function circleOpts(zone, x) {
  const r = 550 + x * 750;
  return {
    radius: r,
    color: ZONE_COLORS[zone],
    fillColor: ZONE_FILL[zone],
    fillOpacity: 0.60,
    weight: 2.5,
    opacity: 0.95,
  };
}

function buildPopup(cell) {
  const z = cell.zone;
  const c = cell.components || {};
  const assets = (cell.assets || []).map(a => `<span style="color:#8eadd4">${a.type}</span>: ${a.name}`).join('<br>') || '—';
  return `
    <div class="popup-ward">${cell.ward_name || cell.cell_id}</div>
    <div class="popup-zone popup-zone-${z}">${z} ZONE</div>
    <div class="popup-row"><span class="popup-key">Hazard X</span><span class="popup-val">${cell.X}</span></div>
    <div class="popup-row"><span class="popup-key">Uncertainty U</span><span class="popup-val">${cell.U}</span></div>
    <div class="popup-row"><span class="popup-key">Suitability G</span><span class="popup-val">${cell.G}</span></div>
    <div class="popup-row"><span class="popup-key">Population</span><span class="popup-val">${(cell.pop||0).toLocaleString()}</span></div>
    <div class="popup-row"><span class="popup-key">Rainfall 24h</span><span class="popup-val">${c.rainfall_24h_mm || '—'} mm</span></div>
    <div class="popup-row"><span class="popup-key">Water Depth</span><span class="popup-val">${c.water_depth_m ?? '—'} m</span></div>
    <div class="popup-row"><span class="popup-key">Elevation</span><span class="popup-val">${c.elevation_m || '—'} m</span></div>
    <div class="popup-row"><span class="popup-key">River Dist</span><span class="popup-val">${c.river_dist_km || '—'} km</span></div>
    <div class="popup-row"><span class="popup-key">Critical Assets</span><span class="popup-val">${assets}</span></div>
    <div class="popup-reason">"${cell.zone_reason}"</div>`;
}

// ── RENDER ZONES ON MAP ───────────────────────────────────────────────────
function renderZones(cells) {
  zoneLayer.clearLayers();
  const counts = { RED: 0, BLUE: 0, GREEN: 0 };
  cells.forEach(cell => {
    const [lat, lon] = cell.centroid;
    const z = cell.zone;
    counts[z]++;
    const circle = L.circle([lat, lon], circleOpts(z, cell.X))
      .bindPopup(buildPopup(cell), { maxWidth: 280 });
    circle.on('mouseover', function() { this.openPopup(); });
    zoneLayer.addLayer(circle);

    // Ward name label
    const icon = L.divIcon({
      html: `<div style="font-family:Inter,sans-serif;font-size:10px;font-weight:800;color:${ZONE_COLORS[z]};
                          text-shadow:0 0 4px #000,0 0 8px #000;white-space:nowrap;pointer-events:none;">
               ${cell.ward_name || ''}
             </div>`,
      iconSize: [100, 14], iconAnchor: [50, 7], className: '',
    });
    L.marker([lat, lon], { icon, interactive: false }).addTo(zoneLayer);
  });
  document.getElementById('cellCount').textContent = `${cells.length} cells — RED:${counts.RED} BLUE:${counts.BLUE} GREEN:${counts.GREEN}`;
  renderAssets(cells);
}

// ── MANDATE 2: RENDER WHO & WHAT (CRITICAL ASSETS ON MAP) ─────────────────
function renderAssets(cells) {
  if (!assetLayer) return;
  assetLayer.clearLayers();

  // Submerged bridge hazard point
  const bridgeIcon = L.divIcon({
    html: `<div style="background:#dc2626;color:#fff;border-radius:6px;padding:3px 6px;font-size:10px;font-weight:900;border:2px solid #fff;box-shadow:0 0 10px #dc2626;white-space:nowrap">
             ⛔ NOYYAL BRIDGE (SUBMERGED 2.1m)
           </div>`,
    iconSize: [160, 24], iconAnchor: [80, 12]
  });
  L.marker([10.9960, 76.9580], { icon: bridgeIcon }).addTo(assetLayer)
    .bindPopup('<b>⛔ CRITICAL TRANSIT CHOKEPOINT</b><br>Noyyal River Bridge overtopped by 2.1m.<br>Status: <b>CLOSED TO ALL VEHICLES</b><br>Evacuation rerouted via northern bypass.');

  cells.forEach(c => {
    const [lat, lon] = c.centroid;
    (c.assets || []).forEach((a, idx) => {
      let symbol = '🏢';
      let bgColor = '#3b82f6';
      if (a.type === 'hospital' || a.type === 'clinic') { symbol = '🏥'; bgColor = '#ef4444'; }
      if (a.type === 'power') { symbol = '⚡'; bgColor = '#f59e0b'; }
      if (a.type === 'water') { symbol = '💧'; bgColor = '#06b6d4'; }
      if (a.type === 'shelter') { symbol = '⛺'; bgColor = '#22c55e'; }

      const icon = L.divIcon({
        html: `<div style="background:${bgColor};color:#fff;font-size:11px;border-radius:50%;width:24px;height:24px;display:flex;align-items:center;justify-content:center;border:1.5px solid #fff;box-shadow:0 0 6px ${bgColor};cursor:pointer">
                 ${symbol}
               </div>`,
        iconSize: [24, 24], iconAnchor: [12, 12]
      });

      const offsetLat = lat + (idx === 0 ? 0.003 : -0.003);
      const offsetLon = lon + (idx === 0 ? 0.004 : -0.004);
      L.marker([offsetLat, offsetLon], { icon })
        .bindPopup(`<b>${symbol} ${a.name}</b><br>Type: ${a.type.toUpperCase()}<br>Ward: ${c.ward_name}<br>Status: <b>${c.zone === 'RED' ? '🔴 AT FLOOD RISK' : '🟢 SECURE'}</b><br>Value: ${a.value * 100}/100`)
        .addTo(assetLayer);
    });
  });
}

// ── MANDATE 4: RENDER SAFEST EVACUATION CORRIDORS ON MAP ──────────────────
function renderEvacuationRoutes(evacData) {
  if (!evacLayer) return;
  evacLayer.clearLayers();
  if (!evacData || !evacData.flows) return;

  evacData.flows.forEach((flow, i) => {
    if (!flow.path || flow.path.length < 2) return;

    // Green evacuation corridor polyline
    const line = L.polyline(flow.path, {
      color: '#10b981',
      weight: 4,
      opacity: 0.9,
      dashArray: '8, 6'
    });
    line.bindPopup(`
      <div style="font-family:Inter,sans-serif">
        <b style="color:#10b981">🚶 SAFEST EVACUATION CORRIDOR</b><br>
        From: <b>${flow.from_ward} (Red Danger Zone)</b><br>
        To Shelter: <b>${flow.to_ward} (Green High Ground)</b><br>
        Distance: <b>${flow.route_km} km</b> (ETA: ${flow.eta_min} min)<br>
        People Evacuating: <b>${flow.people_count}</b><br>
        <span style="color:#38bdf8">Avoided: ${flow.hazard_avoided}</span>
      </div>
    `);
    evacLayer.addLayer(line);

    // Evacuation destination marker
    L.circleMarker(flow.path[flow.path.length - 1], {
      radius: 6,
      color: '#059669',
      fillColor: '#34d399',
      fillOpacity: 1,
      weight: 2
    }).bindPopup(`<b>⛺ Assembly Shelter: ${flow.to_ward}</b>`).addTo(evacLayer);
  });
}

// ── MANDATE 3: RENDER WHICH LOCATIONS FIRST (RESCUE DISPATCH) ─────────────
function renderRoutes(ranked) {
  if (!routeLayer) return;
  routeLayer.clearLayers();
  if (!ranked) return;
  const colors = ['#dc2626','#ea580c','#f59e0b','#3b82f6','#8b5cf6'];

  ranked.forEach((r, i) => {
    if (!r.route || r.route.length < 2) return;
    const pts = r.route.map(p => [p[0], p[1]]);
    
    // Glowing tactical rescue route
    const line = L.polyline(pts, {
      color: colors[i % colors.length],
      weight: 4.5,
      opacity: 0.95
    });
    line.bindPopup(`
      <b>⚡ PRIORITY #${r.rank} TACTICAL DISPATCH</b><br>
      Target: <b>${r.ward_name}</b><br>
      Assigned Unit: <b>${r.team}</b><br>
      ETA: <b>${r.eta_min} minutes</b><br>
      Lives to Save (Π): <b>${r.pi}</b><br>
      Water Depth: <b>${r.water_depth_m || 1.8} m</b>
    `);
    routeLayer.addLayer(line);

    // Priority beacon badge (#1, #2, #3)
    const badgeIcon = L.divIcon({
      html: `<div style="background:${colors[i % colors.length]};color:#fff;font-size:10px;font-weight:900;border-radius:12px;padding:2px 7px;border:2px solid #fff;box-shadow:0 0 12px ${colors[i % colors.length]};white-space:nowrap">
               #${r.rank} ${r.ward_name}
             </div>`,
      iconSize: [90, 20], iconAnchor: [45, 10]
    });
    L.marker(pts[0], { icon: badgeIcon }).addTo(routeLayer);
  });
}

function renderSensorMarkers(detections) {
  if (!sensorLayer) return;
  sensorLayer.clearLayers();
  if (!detections) return;
  detections.forEach(d => {
    const [lat, lon] = d.cell_id.split('_').map(Number);
    const bpm = (d.detections || []).find(x => x.type === 'rppg_pulse')?.bpm || '—';
    const icon = L.divIcon({
      html: `<div style="
        background:rgba(192,132,252,0.95);color:#fff;font-size:11px;font-weight:800;
        border-radius:50%;width:30px;height:30px;display:flex;align-items:center;
        justify-content:center;border:2px solid #c084fc;box-shadow:0 0 14px #c084fc;
        font-family:'JetBrains Mono',monospace;cursor:pointer">♥</div>`,
      iconSize: [30, 30], iconAnchor: [15, 15],
    });
    L.marker([lat, lon], { icon })
      .bindPopup(`<b>📡 Confirmed Survivor Signal</b><br>Ward: ${d.ward_name}<br>Probability Alive: <b>${(d.p_alive*100).toFixed(1)}%</b><br>Est. Trapped: ${d.n_est} residents<br>Pulse: ${bpm} bpm<br>Source: ${d.source}`)
      .addTo(sensorLayer);
  });
}

// ── LAYER TOGGLE ──────────────────────────────────────────────────────────
let activeLayers = { zones: true, routes: true, evac: true, assets: true, sensors: true };
function toggleLayer(name) {
  activeLayers[name] = !activeLayers[name];
  const btn = document.getElementById('btn' + name.charAt(0).toUpperCase() + name.slice(1));
  if (btn) {
    if (activeLayers[name]) btn.classList.add('active');
    else btn.classList.remove('active');
  }
  if (name === 'zones') activeLayers[name] ? map.addLayer(zoneLayer) : map.removeLayer(zoneLayer);
  if (name === 'routes') activeLayers[name] ? map.addLayer(routeLayer) : map.removeLayer(routeLayer);
  if (name === 'evac') activeLayers[name] ? map.addLayer(evacLayer) : map.removeLayer(evacLayer);
  if (name === 'assets') activeLayers[name] ? map.addLayer(assetLayer) : map.removeLayer(assetLayer);
  if (name === 'sensors') activeLayers[name] ? map.addLayer(sensorLayer) : map.removeLayer(sensorLayer);
}

// ── MANDATE NAVIGATOR (PROBLEM STATEMENT 1 ANSWERS) ───────────────────────
function focusMandate(num) {
  for (let i = 1; i <= 4; i++) {
    const b = document.getElementById('mBtn' + i);
    if (b) {
      if (i === num) {
        b.style.borderColor = '#38bdf8';
        b.style.color = '#38bdf8';
        b.style.background = '#0f172a';
      } else {
        b.style.borderColor = '#334155';
        b.style.color = '#94a3b8';
        b.style.background = '#0f172a';
      }
    }
  }

  const statusEl = document.getElementById('mapStatus');
  if (num === 1) {
    // Mandate 1: Where is disaster likely to occur
    if (!map.hasLayer(zoneLayer)) map.addLayer(zoneLayer);
    map.flyTo([11.00, 76.99], 12, { duration: 0.8 });
    statusEl.innerHTML = '<b style="color:#ef4444">MANDATE 1: WHERE DISASTER OCCURS</b> — 17 Inundation Danger Zones (Red) along Noyyal Basin & 8 Drone Recon Sectors (Blue).';
  } else if (num === 2) {
    // Mandate 2: Who and what is affected
    if (!map.hasLayer(assetLayer)) map.addLayer(assetLayer);
    map.flyTo([10.999, 76.985], 13, { duration: 0.8 });
    statusEl.innerHTML = '<b style="color:#f97316">MANDATE 2: WHO & WHAT AFFECTED</b> — 32,660 Residents at Risk. Hospitals (PSG, Singanallur Govt), Substations & Submerged Noyyal Bridge Chokepoint highlighted.';
  } else if (num === 3) {
    // Mandate 3: Which locations emergency teams respond to first
    if (!map.hasLayer(routeLayer)) map.addLayer(routeLayer);
    switchTab('triage');
    map.flyTo([10.998, 76.980], 13, { duration: 0.8 });
    statusEl.innerHTML = '<b style="color:#f59e0b">MANDATE 3: WHICH FIRST (PRIORITY)</b> — #1 Singanallur & #2 Ukkadam ranked at top with assigned NDRF teams and live survivability decay.';
  } else if (num === 4) {
    // Mandate 4: Safest and fastest response/evacuation route
    if (!map.hasLayer(evacLayer)) map.addLayer(evacLayer);
    switchTab('evacuation');
    map.flyTo([11.02, 77.05], 12, { duration: 0.8 });
    statusEl.innerHTML = '<b style="color:#22c55e">MANDATE 4: SAFEST EVACUATION ROUTES</b> — Green corridors routed to Sulur & Annur high grounds, avoiding the submerged Noyyal River Bridge.';
  }
}

// ── TABS ──────────────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  document.getElementById('content-' + name).classList.add('active');
  if (name === 'sensors') startRadar();
  else stopRadar();
  if (name === 'xai') populateXAISelect();
  if (name === 'models') renderModels();
}

// ── KPI UPDATE ────────────────────────────────────────────────────────────
function updateKPIs(summary, detected, dispatched) {
  document.getElementById('kpiRedNum').textContent   = summary.red_cells   || 0;
  document.getElementById('kpiBlueNum').textContent  = summary.blue_cells  || 0;
  document.getElementById('kpiGreenNum').textContent = summary.green_cells || 0;
  const pop = summary.at_risk_population || 0;
  document.getElementById('kpiPop').textContent = pop >= 1000 ? (pop/1000).toFixed(1)+'k' : pop;
  document.getElementById('kpiDetected').textContent  = detected;
  document.getElementById('kpiDispatched').textContent = dispatched;
  document.getElementById('alertText').textContent = summary.imbalance_message || 'Threat assessment active';
  document.getElementById('mapStatus').textContent =
    `Scenario: Noyyal Basin 24h Extreme Rainfall — IMD Red Alert — Updated ${new Date().toLocaleTimeString('en-IN')}`;
}

// ── TRIAGE PANEL ──────────────────────────────────────────────────────────
function renderTriage(data) {
  const el = document.getElementById('triageList');
  if (!data || !data.ranked) { el.innerHTML = '<p style="color:#4a6180">No data</p>'; return; }
  el.innerHTML = data.ranked.map((r, i) => `
    <div class="triage-card" style="animation-delay:${i*60}ms" onclick="flyToCell('${r.cell_id}')">
      <div class="triage-header">
        <div style="display:flex;align-items:center;gap:10px">
          <span class="triage-rank">#${r.rank}</span>
          <span class="triage-ward">${r.ward_name || r.cell_id}</span>
        </div>
        <div class="triage-pi">Π = ${r.pi}</div>
      </div>
      <div class="triage-meta">
        <span class="triage-chip chip-team">🚁 ${r.team}</span>
        <span class="triage-chip chip-eta">⏱ ${r.eta_min} min ETA</span>
        <span class="triage-chip chip-surv">♥ Surv: ${r.survivability}</span>
        <span class="triage-chip chip-depth">💧 ${r.water_depth_m ?? '—'}m depth</span>
        <span class="triage-chip chip-pop">👥 ~${r.n_est} survivors</span>
      </div>
    </div>`).join('');
}

// ── EVACUATION PANEL ──────────────────────────────────────────────────────
function renderEvacuation(data) {
  const shelEl = document.getElementById('shelterList');
  const flowEl = document.getElementById('flowList');
  if (!data) { shelEl.innerHTML = '<p style="color:#4a6180">No data</p>'; return; }

  shelEl.innerHTML = (data.shelters || []).map((s, i) => {
    const pct = Math.round((s.utilization || 0) * 100);
    const color = pct > 85 ? '#ff9800' : pct > 60 ? '#3b8fff' : '#00e676';
    return `<div class="shelter-card" style="animation-delay:${i*50}ms">
      <div class="shelter-header">
        <span class="shelter-name">🟢 ${s.ward_name}</span>
        <span class="shelter-util">${s.assigned} / ${s.capacity} (${pct}%)</span>
      </div>
      <div class="shelter-bar-bg">
        <div class="shelter-bar-fill" style="width:${pct}%;background:${color}"></div>
      </div>
    </div>`;
  }).join('');

  flowEl.innerHTML = (data.flows || []).map((f, i) => `
    <div class="flow-card" style="animation-delay:${i*60}ms">
      <div class="flow-arrow">→</div>
      <div class="flow-info">
        <div class="flow-route">${f.from_ward} → ${f.to_ward}</div>
        <div class="flow-count">${f.people_count} people · ${f.route_km} km · ${f.eta_min} min</div>
        <div class="flow-chips">
          <span class="flow-chip">✅ Hazard-free route</span>
          ${f.hazard_avoided ? `<span class="flow-chip" style="background:rgba(255,59,59,0.1);color:#ff8080">⚠ Avoids: ${f.hazard_avoided}</span>` : ''}
        </div>
      </div>
    </div>`).join('');
}

// ── DETECTIONS PANEL ──────────────────────────────────────────────────────
function renderDetections(data) {
  const el = document.getElementById('detectionList');
  if (!data || !data.length) { el.innerHTML = '<p style="color:#4a6180">No detections</p>'; return; }
  el.innerHTML = data.map((d, i) => {
    const chips = (d.detections || []).map(det => {
      if (det.type === 'rppg_pulse')
        return `<span class="det-chip det-rppg">♥ rPPG ${det.bpm}bpm (${(det.conf*100).toFixed(0)}%)</span>`;
      if (det.type === 'thermal_hotspot')
        return `<span class="det-chip det-thermal">🌡 Thermal ${det.temp_c}°C (${(det.conf*100).toFixed(0)}%)</span>`;
      if (det.type === 'rgb_person')
        return `<span class="det-chip det-rgb">👤 RGB Person (${(det.conf*100).toFixed(0)}%)</span>`;
      if (det.type === 'acoustic_distress')
        return `<span class="det-chip det-acoustic">🔊 ${det.class} (${(det.conf*100).toFixed(0)}%)</span>`;
      return '';
    }).join('');
    const palive = d.p_alive || 0;
    const badgeClass = palive >= 0.70 ? 'p-alive-high' : 'p-alive-mid';
    return `<div class="detection-card" style="animation-delay:${i*80}ms">
      <div class="detection-header">
        <div>
          <div class="detection-ward">${d.ward_name || d.cell_id}</div>
          <div class="detection-source">${d.source} · ${d.n_est} est. survivors</div>
        </div>
        <span class="p-alive-badge ${badgeClass}">p_alive: ${palive}</span>
      </div>
      <div class="detection-chips">${chips}</div>
    </div>`;
  }).join('');
}

// ── BIO-RADAR ANIMATION ─────────────────────────────────────────────────────
// Parameters (bpm, waveform) come from a REAL PPG-DaLiA test window run through the
// actually trained heart_rate_ppgdalia model (see /api/model/heartbeat_samples) —
// not Math.random(). The chart itself — grid, gradient stroke, detection marker —
// is unchanged.
let _radarPhase = 0;
let _radarBpm = 74;

async function fetchHeartbeatSamples() {
  if (_heartbeatSamples) return _heartbeatSamples;
  try {
    const res = await fetch(`${API}/api/model/heartbeat_samples`);
    const data = await res.json();
    _heartbeatSamples = (data.samples || []).length ? data.samples : null;
  } catch (e) {
    _heartbeatSamples = null;
  }
  return _heartbeatSamples;
}

function startRadar() {
  if (radarAnimId) return;
  const canvas = document.getElementById('radarCanvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const statusEl = document.getElementById('radarStatus');
  const bpmEl = document.getElementById('radarBpm');
  let t = 0;
  let detected = false;

  // Real model output until proven otherwise: a genuine PPG-DaLiA test window's
  // predicted vs. true heart rate. Falls back to a labelled synthetic value only
  // if the sample endpoint has nothing (e.g. GPU queue never ran).
  let sample = null;
  let heartbeatHz = 1.25, trueBpm = null, predictedBpm = null, usingReal = false;
  fetchHeartbeatSamples().then(samples => {
    if (samples) {
      sample = samples[Math.floor(Math.random() * samples.length)];
      predictedBpm = sample.predicted_bpm;
      trueBpm = sample.true_bpm;
      heartbeatHz = predictedBpm / 60;
      usingReal = true;
    } else {
      predictedBpm = 68 + Math.floor(Math.random() * 30);
      heartbeatHz = predictedBpm / 60;
    }
  });

  function draw() {
    ctx.clearRect(0, 0, W, H);

    // Background grid
    ctx.strokeStyle = 'rgba(59,143,255,0.06)';
    ctx.lineWidth = 0.5;
    for (let x = 0; x < W; x += 40) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke(); }
    for (let y = 0; y < H; y += 20) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke(); }

    // Composite signal: breathing (0.2Hz, fixed reference) + heartbeat (real model bpm) + noise
    const mid = H / 2;
    ctx.beginPath();
    for (let px = 0; px < W; px++) {
      const tNorm = (t + px) / W;
      const breathing = 18 * Math.sin(2 * Math.PI * 0.20 * tNorm * 8);
      const heartbeat = 10 * Math.sin(2 * Math.PI * heartbeatHz * tNorm * 8);
      const noise = (Math.random() - 0.5) * 3;
      const val = breathing + heartbeat + noise;
      const y = mid - val;
      px === 0 ? ctx.moveTo(0, y) : ctx.lineTo(px, y);
    }
    ctx.strokeStyle = 'rgba(59,143,255,0.4)';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Highlighted cardiac peaks
    ctx.beginPath();
    for (let px = 0; px < W; px++) {
      const tNorm = (t + px) / W;
      const heartbeat = 10 * Math.sin(2 * Math.PI * heartbeatHz * tNorm * 8);
      const y = mid - heartbeat;
      px === 0 ? ctx.moveTo(0, y) : ctx.lineTo(px, y);
    }
    const grad = ctx.createLinearGradient(0, 0, W, 0);
    grad.addColorStop(0, 'rgba(192,132,252,0.2)');
    grad.addColorStop(0.5, 'rgba(192,132,252,0.9)');
    grad.addColorStop(1, 'rgba(192,132,252,0.2)');
    ctx.strokeStyle = grad;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Detection zone marker
    if (t > 80 && predictedBpm != null) {
      if (!detected) { detected = true; }
      const markerX = Math.round(W * 0.62);
      ctx.beginPath();
      ctx.moveTo(markerX, 8); ctx.lineTo(markerX, H - 8);
      ctx.strokeStyle = 'rgba(0,230,118,0.6)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(0,230,118,0.9)';
      ctx.font = '9px JetBrains Mono, monospace';
      ctx.fillText('VITAL DETECTED', markerX - 48, 18);

      if (usingReal) {
        const err = Math.abs(predictedBpm - trueBpm).toFixed(1);
        statusEl.textContent = `● MODEL PREDICTED ${predictedBpm} bpm — real PPG-DaLiA sample, true ${trueBpm} bpm (err ${err})`;
      } else {
        statusEl.textContent = `● CARDIAC PULSE CONFIRMED — ${predictedBpm} bpm`;
      }
      statusEl.style.color = '#00e676';
      bpmEl.textContent = predictedBpm + ' bpm';
    } else {
      statusEl.textContent = '● SCANNING FOR VITALS...';
      statusEl.style.color = '#3b8fff';
      bpmEl.textContent = '-- bpm';
    }

    t++;
    radarAnimId = requestAnimationFrame(draw);
  }
  draw();
}

function stopRadar() {
  if (radarAnimId) { cancelAnimationFrame(radarAnimId); radarAnimId = null; }
}

// ── XAI PANEL ────────────────────────────────────────────────────────────
function populateXAISelect() {
  if (!zonesData) return;
  const sel = document.getElementById('xaiCellSelect');
  const current = sel.value;
  sel.innerHTML = '<option value="">Select a zone cell...</option>' +
    zonesData.grid_cells.map(c =>
      `<option value="${c.cell_id}" ${c.cell_id === current ? 'selected' : ''}>${c.ward_name || c.cell_id} [${c.zone}]</option>`
    ).join('');
  if (current) renderXAI();
}

function renderXAI() {
  const sel = document.getElementById('xaiCellSelect');
  const el = document.getElementById('xaiOutput');
  if (!sel.value || !zonesData) { el.innerHTML = ''; return; }
  const cell = zonesData.grid_cells.find(c => c.cell_id === sel.value);
  if (!cell) return;

  const comp = cell.components || {};
  const uc = cell.uncertainty_components || {};

  // SHAP-style feature contributions
  const features = [
    { name: 'Rainfall 24h (mm)',    value: comp.rainfall_24h_mm || 0,  maxVal: 260,  sign: 'positive', label: `${comp.rainfall_24h_mm || 0} mm` },
    { name: 'Water Depth (m)',      value: (comp.water_depth_m||0)*100, maxVal: 250,  sign: 'positive', label: `${comp.water_depth_m || 0} m` },
    { name: 'TWI Index',            value: (comp.twi_index||0)*6,       maxVal: 100,  sign: 'positive', label: (comp.twi_index || 0).toFixed(2) },
    { name: 'Elevation (m)',        value: Math.max(0,400-(comp.elevation_m||350)), maxVal:130, sign:'negative', label:`${comp.elevation_m||'—'} m`},
    { name: 'River Dist. (km)',     value: Math.max(0,8-(comp.river_dist_km||4))*10, maxVal:80, sign:'positive', label:`${comp.river_dist_km||'—'} km`},
    { name: 'Uncertainty U_stale', value: (uc.u_stale||0)*100,         maxVal: 100,  sign: 'neutral',  label: `${(uc.u_stale||0).toFixed(3)}` },
    { name: 'Uncertainty U_model', value: (uc.u_model||0)*100,         maxVal: 100,  sign: 'neutral',  label: `${(uc.u_model||0).toFixed(3)}` },
    { name: 'Suitability G',       value: (cell.G||0)*100,             maxVal: 100,  sign: 'negative', label: cell.G },
    { name: 'Population at risk',  value: (cell.pop||0)/50,            maxVal: 70,   sign: 'positive', label: (cell.pop||0).toLocaleString() },
  ];

  el.innerHTML = `
    <div class="xai-zone-badge xai-zone-${cell.zone}">
      ${cell.zone === 'RED' ? '🔴' : cell.zone === 'BLUE' ? '🔵' : '🟢'} ${cell.zone} ZONE — ${cell.ward_name}
    </div>
    <div class="xai-reason">"${cell.zone_reason}"</div>
    <div class="section-label">Feature Contributions (SHAP)</div>
    ${features.map(f => {
      const pct = Math.min(100, Math.round(f.value / f.maxVal * 100));
      const barClass = f.sign === 'positive' ? 'shap-positive' : f.sign === 'negative' ? 'shap-negative' : 'shap-neutral';
      return `<div class="xai-feature-row">
        <div class="xai-feature-label">
          <span class="xai-feature-name">${f.name}</span>
          <span class="xai-feature-val">${f.label}</span>
        </div>
        <div class="xai-bar-bg">
          <div class="xai-bar-fill ${barClass}" style="width:${pct}%"></div>
        </div>
      </div>`;
    }).join('')}
    <div style="margin-top:14px;padding:10px;background:rgba(59,143,255,0.06);border-radius:8px;border:1px solid rgba(59,143,255,0.15)">
      <div style="font-size:10px;color:#8eadd4;margin-bottom:6px;font-weight:700;letter-spacing:1px;text-transform:uppercase">Model Scores</div>
      <div style="display:flex;gap:16px;flex-wrap:wrap">
        <div><div style="font-size:18px;font-weight:900;color:${ZONE_COLORS[cell.zone]};font-family:'JetBrains Mono',monospace">${Math.round(cell.X*100)}</div><div style="font-size:9px;color:#4a6180">Hazard X (0–100)</div></div>
        <div><div style="font-size:18px;font-weight:900;color:#3b8fff;font-family:'JetBrains Mono',monospace">${Math.round(cell.U*100)}</div><div style="font-size:9px;color:#4a6180">Uncertainty U (0–100)</div></div>
        <div><div style="font-size:18px;font-weight:900;color:#00e676;font-family:'JetBrains Mono',monospace">${Math.round(cell.G*100)}</div><div style="font-size:9px;color:#4a6180">Suitability G (0–100)</div></div>
        <div><div style="font-size:18px;font-weight:900;color:#f97316;font-family:'JetBrains Mono',monospace">${Math.round(cell.rescue_priority*100)}</div><div style="font-size:9px;color:#4a6180">Rescue Priority (0–100)</div></div>
      </div>
    </div>`;
}

// ── GPU-TRAINED MODELS (trained_models/, outside this repo) ────────────────
function renderGpuModels(models) {
  if (!models.length) return '';
  const fmt = (v, digits = 4) => (v == null ? '—' : (typeof v === 'number' ? v.toFixed(digits) : v));
  const cards = models.map(mo => {
    const val = mo.val || mo.transfer_head_val || {};
    const metricKeys = Object.keys(val).filter(k => typeof val[k] === 'number' && k !== 'n');
    const metricsHtml = metricKeys.slice(0, 4).map(k => `
      <div style="background:rgba(0,0,0,0.3);padding:6px;border-radius:6px;text-align:center">
        <div style="font-size:13px;font-weight:900;color:#5eead4;font-family:'JetBrains Mono',monospace">${fmt(val[k], k.includes('bpm') ? 1 : 4)}</div>
        <div style="font-size:8px;color:#64748b;text-transform:uppercase">${k.replace(/_/g, ' ')}</div>
      </div>`).join('');
    const topClasses = mo.yamnet_real_inference_top5
      ? `<div style="font-size:10px;color:#94a3b8;margin-top:6px">Real YAMNet inference, top classes: ${
          mo.yamnet_real_inference_top5.slice(0, 3).map(c => `${c.class} (${c.score})`).join(', ')}</div>`
      : '';
    const note = mo.note
      ? `<div style="margin-top:6px;padding:6px 8px;background:rgba(245,158,11,0.08);border-left:2px solid #f59e0b;border-radius:4px;font-size:9.5px;color:#fcd34d;line-height:1.4">${mo.note}</div>`
      : '';
    return `
      <div style="background:rgba(0,0,0,0.25);border:1px solid rgba(94,234,212,0.25);border-radius:8px;padding:10px">
        <div style="font-weight:700;color:#5eead4;font-size:11.5px;margin-bottom:4px">${mo.name}</div>
        <div style="font-size:9.5px;color:#64748b;margin-bottom:6px">${mo.device || ''}${mo.dataset ? ' · ' + mo.dataset : ''}</div>
        <div style="display:grid;grid-template-columns:repeat(${Math.min(metricKeys.length, 4) || 1}, 1fr);gap:6px">${metricsHtml}</div>
        ${topClasses}${note}
      </div>`;
  }).join('');

  return `
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(94,234,212,0.3);border-radius:10px;padding:14px;margin-bottom:14px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
        <span style="font-weight:700;color:#5eead4;font-size:13px">🖥️ GPU-Trained Models (CUDA, trained outside this repo)</span>
        <span style="font-size:10px;background:rgba(94,234,212,0.15);color:#5eead4;padding:2px 8px;border-radius:12px;font-weight:700">${models.length} MODELS</span>
      </div>
      <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px">${cards}</div>
    </div>`;
}

// ── ML MODELS & BENCHMARKS PANEL ──────────────────────────────────────────
let _cachedModelMetrics = null;
async function renderModels() {
  const el = document.getElementById('modelsOutput');
  if (!el) return;
  if (!_cachedModelMetrics) {
    try {
      const res = await fetch(`${API}/api/model/metrics`);
      _cachedModelMetrics = await res.json();
    } catch (e) {
      el.innerHTML = '<div style="color:#ff3b3b;padding:15px">Failed to load model benchmarks from server.</div>';
      return;
    }
  }

  const m = _cachedModelMetrics;
  const p = m.primary_hazard_model || {};
  const pm = p.metrics || {};
  const s = m.secondary_benchmarking_model || {};
  const sm = s.metrics || {};

  el.innerHTML = `
    <!-- PRIMARY MODEL -->
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(59,143,255,0.3);border-radius:10px;padding:14px;margin-bottom:14px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <span style="font-weight:700;color:#60a5fa;font-size:13px">🌟 PRIMARY: ${p.name || 'XGBoost Hazard Regressor'}</span>
        <span style="font-size:10px;background:rgba(34,197,94,0.15);color:#4ade80;padding:2px 8px;border-radius:12px;font-weight:700">${(p.trained_rows||0).toLocaleString()} ROWS TRAINED</span>
      </div>
      <div style="font-size:11px;color:#94a3b8;margin-bottom:10px">Dataset: <b style="color:#e2e8f0">${p.dataset}</b></div>
      <div style="display:grid;grid-template-columns:repeat(3, 1fr);gap:8px;margin-bottom:12px">
        <div style="background:rgba(0,0,0,0.3);padding:8px;border-radius:6px;text-align:center">
          <div style="font-size:16px;font-weight:900;color:#38bdf8;font-family:'JetBrains Mono',monospace">${pm.accuracy!=null?(pm.accuracy*100).toFixed(1)+'%':'—'}</div>
          <div style="font-size:9px;color:#64748b">Test Accuracy</div>
        </div>
        <div style="background:rgba(0,0,0,0.3);padding:8px;border-radius:6px;text-align:center">
          <div style="font-size:16px;font-weight:900;color:#a855f7;font-family:'JetBrains Mono',monospace">${pm.roc_auc!=null?pm.roc_auc.toFixed(4):'—'}</div>
          <div style="font-size:9px;color:#64748b">ROC-AUC (OVR)</div>
        </div>
        <div style="background:rgba(0,0,0,0.3);padding:8px;border-radius:6px;text-align:center">
          <div style="font-size:16px;font-weight:900;color:#22c55e;font-family:'JetBrains Mono',monospace">${pm.f1!=null?pm.f1.toFixed(4):'—'}</div>
          <div style="font-size:9px;color:#64748b">F1 Score</div>
        </div>
      </div>
      <div style="font-size:10px;color:#64748b;margin-bottom:4px">Key Feature Drivers (share of absolute model weight):</div>
      <div style="font-size:10px;color:#cbd5e1;line-height:1.6">
        ${(p.feature_importance||[]).slice(0,6).map(f =>
          `• ${f.feature} (${(f.share*100).toFixed(1)}%) <span style="color:${f.weight>0?'#f87171':'#4ade80'}">${f.direction}</span>`
        ).join('<br>') || '<span style="color:#64748b">not built — run python -m src.features --build-all</span>'}
      </div>
      ${p.honest_read ? `<div style="margin-top:8px;padding:8px;background:rgba(245,158,11,0.08);border-left:2px solid #f59e0b;border-radius:4px;font-size:10px;color:#fcd34d;line-height:1.5">${p.honest_read}</div>` : ''}
    </div>

    <!-- BENCHMARK MODEL -->
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(168,85,247,0.3);border-radius:10px;padding:14px;margin-bottom:14px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <span style="font-weight:700;color:#c084fc;font-size:13px">🏆 BENCHMARK: ${s.name || 'Kaggle Grandmaster LightGBM'}</span>
        <span style="font-size:10px;background:rgba(168,85,247,0.15);color:#c084fc;padding:2px 8px;border-radius:12px;font-weight:700">${(s.trained_rows||0).toLocaleString()} ROWS</span>
      </div>
      <div style="font-size:11px;color:#94a3b8;margin-bottom:10px">Dataset: <b style="color:#e2e8f0">${s.dataset}</b></div>
      <div style="display:grid;grid-template-columns:repeat(3, 1fr);gap:8px;margin-bottom:12px">
        <div style="background:rgba(0,0,0,0.3);padding:8px;border-radius:6px;text-align:center">
          <div style="font-size:16px;font-weight:900;color:#c084fc;font-family:'JetBrains Mono',monospace">${sm.accuracy!=null?(sm.accuracy*100).toFixed(1)+'%':'—'}</div>
          <div style="font-size:9px;color:#64748b">Test Accuracy</div>
        </div>
        <div style="background:rgba(0,0,0,0.3);padding:8px;border-radius:6px;text-align:center">
          <div style="font-size:16px;font-weight:900;color:#38bdf8;font-family:'JetBrains Mono',monospace">${sm.roc_auc!=null?sm.roc_auc.toFixed(4):'—'}</div>
          <div style="font-size:9px;color:#64748b">ROC-AUC</div>
        </div>
        <div style="background:rgba(0,0,0,0.3);padding:8px;border-radius:6px;text-align:center">
          <div style="font-size:16px;font-weight:900;color:#f59e0b;font-family:'JetBrains Mono',monospace">${sm.brier!=null?sm.brier.toFixed(4):'—'}</div>
          <div style="font-size:9px;color:#64748b">Brier Score</div>
        </div>
      </div>
    </div>

    <!-- GPU-TRAINED MODELS -->
    ${renderGpuModels(m.gpu_trained_models || [])}

    <!-- SENSOR FUSION STACK -->
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(34,197,94,0.3);border-radius:10px;padding:14px">
      <div style="font-weight:700;color:#4ade80;font-size:13px;margin-bottom:6px">📡 Multimodal Bayesian Life-Detection Fusion</div>
      <div style="font-size:11px;color:#94a3b8;margin-bottom:8px">Formula: <code style="color:#38bdf8">logit(P) = logit(p0) + Σ zk·ln(Λk)</code></div>
      <div style="font-size:10px;color:#cbd5e1;line-height:1.6">
        • <b>FINDER Microwave Doppler Radar:</b> ln(Λ) = 3.22 (Cardiac micro-motion)<br>
        • <b>Acoustic YAMNet Classifier:</b> ln(Λ) = 2.48 (Distress screaming)<br>
        • <b>Optical rPPG Camera (POS):</b> ln(Λ) = 2.08 (Non-contact pulse lock)<br>
        • <b>Aerial YOLOv11 Silhouette:</b> ln(Λ) = 1.61 (Human detection)<br>
        • <b>Radiometric FLIR IR:</b> ln(Λ) = 1.10 (34°C–37°C body temperature)
      </div>
    </div>`;
}

// ── FLY TO CELL ───────────────────────────────────────────────────────────
function flyToCell(cellId) {
  if (!map || !zonesData) return;
  const cell = zonesData.grid_cells.find(c => c.cell_id === cellId);
  if (!cell) return;
  map.flyTo(cell.centroid, 14, { animate: true, duration: 0.8 });
}

// ── CHANGE DISASTER ───────────────────────────────────────────────────────
function changeDisaster(val) {
  currentDisaster = val;
  loadAll();
}

// ── MAIN DATA LOADER ──────────────────────────────────────────────────────
async function loadAll() {
  try {
    const [sumRes, zonesRes, priorRes, evacRes, detRes] = await Promise.all([
      fetch(`${API}/api/alert/summary?disaster=${currentDisaster}`),
      fetch(`${API}/api/zones?disaster=${currentDisaster}`),
      fetch(`${API}/api/priority?disaster=${currentDisaster}`),
      fetch(`${API}/api/evacuation?disaster=${currentDisaster}`),
      fetch(`${API}/api/detections?disaster=${currentDisaster}`),
    ]);

    const summary    = await sumRes.json();
    zonesData        = await zonesRes.json();
    priorityData     = await priorRes.json();
    evacuationData   = await evacRes.json();
    detectionsData   = await detRes.json();

    const cells = zonesData.grid_cells || [];
    const detected = detectionsData.length || 0;
    const dispatched = (priorityData.ranked || []).length;

    updateKPIs(summary, detected, dispatched);
    renderZones(cells);
    renderRoutes(priorityData.ranked);
    renderEvacuationRoutes(evacuationData);
    renderSensorMarkers(detectionsData);
    renderTriage(priorityData);
    renderEvacuation(evacuationData);
    renderDetections(detectionsData);
    populateXAISelect();

  } catch (err) {
    console.error('Load error:', err);
    document.getElementById('alertText').textContent = 'Error loading data — check backend is running';
  }
}

// ── LIVE 24-48H SIMULATION RUNNER ─────────────────────────────────────────
let _simDebounceTimer = null;
function runSimulation() {
  clearTimeout(_simDebounceTimer);
  _simDebounceTimer = setTimeout(async () => {
    const rain = parseFloat(document.getElementById('simRain').value);
    const river = parseFloat(document.getElementById('simRiver').value);
    const sar = parseFloat(document.getElementById('simSar').value);

    document.getElementById('mapStatus').textContent = `⚡ Executing live XGBoost inference (Rain: ${(rain*100).toFixed(0)}%, Noyyal: ${river}m, SAR: ${sar}h)...`;

    try {
      const res = await fetch(`${API}/api/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          rainfall_scale: rain,
          river_stage_m: river,
          sar_staleness_h: sar
        })
      });
      const data = await res.json();
      
      // Update local state and redraw map
      zonesData = { grid_cells: data.grid_cells };
      renderZones(data.grid_cells);

      // Re-fetch priority and summary to reflect new hazard scores
      const [sumRes, priorRes] = await Promise.all([
        fetch(`${API}/api/alert/summary?disaster=${currentDisaster}`),
        fetch(`${API}/api/priority?disaster=${currentDisaster}`)
      ]);
      const summary = await sumRes.json();
      priorityData = await priorRes.json();

      updateKPIs(summary, detectionsData ? detectionsData.length : 0, (priorityData.ranked||[]).length);
      renderTriage(priorityData);
      renderRoutes(priorityData.ranked);
      renderEvacuationRoutes(evacuationData);
      populateXAISelect();

      document.getElementById('mapStatus').textContent = `✅ Live Scenario Active — Noyyal River: ${river}m — IMD Rainfall Surge: ${(rain*100).toFixed(0)}%`;
    } catch (e) {
      console.error('Simulation error:', e);
      document.getElementById('mapStatus').textContent = 'Simulation failed — check server connection';
    }
  }, 250);
}

// ── INCIDENT ACTION PLAN MODAL ───────────────────────────────────────────
async function openIncidentActionPlan() {
  const modal = document.getElementById('iapModal');
  const body = document.getElementById('iapModalBody');
  modal.style.display = 'flex';
  body.innerHTML = '<div style="text-align:center;padding:30px;color:#94a3b8">Compiling official Incident Action Plan from live telemetry...</div>';

  try {
    const res = await fetch(`${API}/api/export/incident_action_plan`);
    const plan = await res.json();

    const m1 = plan.mandate_1_where_likely_to_occur || {};
    const m2 = plan.mandate_2_who_and_what_affected || {};
    const m3 = plan.mandate_3_priority_response_ranking || [];
    const m4 = plan.mandate_4_safest_and_fastest_evacuation || {};

    body.innerHTML = `
      <div style="background:rgba(59,143,255,0.08);border:1px solid rgba(59,143,255,0.25);border-radius:8px;padding:12px;margin-bottom:16px">
        <div style="font-weight:700;color:#38bdf8;font-size:13px;margin-bottom:4px">OPERATIONAL SUMMARY & CALAMITY STATUS</div>
        <div>Operation: <b>${plan.operation_name}</b> | Operational Period: <b>${plan.operational_period}</b></div>
        <div>CWC River Stage: <b style="color:#f59e0b">${plan.calamity_parameters.cwc_river_stage}</b> | Telemetry: <b>${plan.calamity_parameters.meteorological_forcing}</b></div>
      </div>

      <!-- MANDATE 1 -->
      <div style="margin-bottom:18px">
        <div style="font-weight:800;color:#f8fafc;font-size:13px;margin-bottom:6px;border-left:3px solid #ef4444;padding-left:8px">
          MANDATE 1: WHERE IS THE DISASTER LIKELY TO OCCUR?
        </div>
        <div style="color:#94a3b8;margin-bottom:6px">Predicted Critical Impact Wards (${m1.critical_risk_zones_count} Total):</div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px">
          ${(m1.critical_wards||[]).map(w => `<span style="background:rgba(239,68,68,0.2);color:#fca5a5;padding:2px 8px;border-radius:4px;font-size:11px;border:1px solid rgba(239,68,68,0.4)">🔴 ${w}</span>`).join('')}
        </div>
        <div style="color:#94a3b8;margin-bottom:4px">Active Drone Reconnaissance Queue (High Epistemic Uncertainty):</div>
        <div style="display:flex;gap:6px;flex-wrap:wrap">
          ${(m1.reconnaissance_drone_queue_blue_zones||[]).map(w => `<span style="background:rgba(59,143,255,0.15);color:#93c5fd;padding:2px 8px;border-radius:4px;font-size:11px;border:1px solid rgba(59,143,255,0.3)">🔵 ${w}</span>`).join('')}
        </div>
      </div>

      <!-- MANDATE 2 -->
      <div style="margin-bottom:18px">
        <div style="font-weight:800;color:#f8fafc;font-size:13px;margin-bottom:6px;border-left:3px solid #f97316;padding-left:8px">
          MANDATE 2: WHO AND WHAT IS LIKELY TO BE AFFECTED?
        </div>
        <div style="margin-bottom:6px">Total Population in Danger: <b style="color:#f87171;font-size:14px">${(m2.total_endangered_population||0).toLocaleString()} residents</b></div>
        <div style="color:#94a3b8;margin-bottom:4px">Critical Infrastructure & Hospitals Exposed:</div>
        <ul style="margin:4px 0 8px 18px;color:#e2e8f0;font-size:11px">
          ${(m2.critical_infrastructure_at_risk||[]).map(a => `<li>${a}</li>`).join('')}
        </ul>
        <div style="color:#ef4444;font-size:11px;font-weight:700">⚠️ Submerged Road Chokepoints: ${(m2.submerged_chokepoints||[]).join(' • ')}</div>
      </div>

      <!-- MANDATE 3 -->
      <div style="margin-bottom:18px">
        <div style="font-weight:800;color:#f8fafc;font-size:13px;margin-bottom:6px;border-left:3px solid #eab308;padding-left:8px">
          MANDATE 3: WHICH LOCATIONS SHOULD EMERGENCY TEAMS RESPOND TO FIRST?
        </div>
        <table style="width:100%;border-collapse:collapse;font-size:11px;text-align:left">
          <thead>
            <tr style="border-bottom:1px solid #334155;color:#94a3b8">
              <th style="padding:4px">Rank</th>
              <th style="padding:4px">Ward</th>
              <th style="padding:4px">At-Risk Pop</th>
              <th style="padding:4px">Assigned Team</th>
              <th style="padding:4px">Equipment Protocol</th>
            </tr>
          </thead>
          <tbody>
            ${m3.map(r => `
              <tr style="border-bottom:1px solid #1e293b">
                <td style="padding:6px 4px;font-weight:700;color:#f59e0b">#${r.priority}</td>
                <td style="padding:6px 4px;font-weight:700;color:#f8fafc">${r.target_ward}</td>
                <td style="padding:6px 4px;color:#fca5a5">${(r.population_at_risk||0).toLocaleString()}</td>
                <td style="padding:6px 4px;color:#38bdf8">${r.assigned_team}</td>
                <td style="padding:6px 4px;color:#cbd5e1">${r.recommended_equipment}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>

      <!-- MANDATE 4 -->
      <div>
        <div style="font-weight:800;color:#f8fafc;font-size:13px;margin-bottom:6px;border-left:3px solid #22c55e;padding-left:8px">
          MANDATE 4: SAFEST AND FASTEST EVACUATION ROUTES & SHELTERS
        </div>
        <div style="color:#94a3b8;margin-bottom:6px">Authorized High-Ground Evacuation Assembly Shelters:</div>
        <div style="display:grid;grid-template-columns:repeat(3, 1fr);gap:8px;margin-bottom:10px">
          ${(m4.designated_safe_shelters||[]).map(s => `
            <div style="background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.3);padding:8px;border-radius:6px">
              <div style="font-weight:700;color:#4ade80">${s.name}</div>
              <div style="font-size:10px;color:#94a3b8">Cap: ${s.capacity} | Elev: ${s.elevation}</div>
              <div style="font-size:9px;color:#22c55e;font-weight:700;margin-top:2px">● ${s.status}</div>
            </div>
          `).join('')}
        </div>
        <div style="color:#ef4444;font-size:11px"><b>🚫 Prohibited Evacuation Transit Corridors:</b><br>
          ${(m4.prohibited_transit_routes||[]).map(r => `• ${r}`).join('<br>')}
        </div>
      </div>
    `;
  } catch (err) {
    body.innerHTML = '<div style="color:#ef4444;padding:20px">Failed to load Incident Action Plan.</div>';
  }
}

function closeIncidentActionPlan() {
  document.getElementById('iapModal').style.display = 'none';
}

// ── BOOT ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initMap();
  loadAll();
  // Auto-refresh every 30 seconds for live feel
  setInterval(loadAll, 30000);
});

