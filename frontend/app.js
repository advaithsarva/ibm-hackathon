/* ═══════════════════════════════════════════════════════════════════════
   The Sentinel Grid — Dashboard Application Logic
   Connects to FastAPI backend, renders Leaflet map, manages panels
   ═══════════════════════════════════════════════════════════════════════ */

const API = 'http://localhost:8000';

// ── Colour maps ────────────────────────────────────────────────────────────
const ZONE_COLORS = {
  RED:   { fill: '#da1e28', stroke: '#ff8389', glow: 'rgba(218,30,40,0.5)' },
  BLUE:  { fill: '#0043ce', stroke: '#78a9ff', glow: 'rgba(0,67,206,0.5)' },
  GREEN: { fill: '#198038', stroke: '#6fdc8c', glow: 'rgba(36,161,72,0.45)' },
};

// ── State ──────────────────────────────────────────────────────────────────
let mapInstance = null;
let zoneLayerGroup = null;
let evacuationLayerGroup = null;
let selectedCell = null;
let radarChart = null;
let gaugeChart = null;
let radarInterval = null;
let currentDisaster = 'flood';
let allZonesData = null;

// ── Clock ──────────────────────────────────────────────────────────────────
function updateClock() {
  const el = document.getElementById('live-clock');
  if (el) el.textContent = new Date().toLocaleTimeString('en-IN', { hour12: false });
}
setInterval(updateClock, 1000);
updateClock();

// ── Init ───────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', async () => {
  initMap();
  await loadAll();
  document.getElementById('disaster-select').addEventListener('change', async (e) => {
    currentDisaster = e.target.value;
    await loadAll();
  });
});

async function loadAll() {
  try {
    await Promise.all([loadSummary(), loadZones(), loadPriority(), loadEvacuation(), loadDetections()]);
  } catch (err) {
    console.error('API error — is the backend running on port 8000?', err);
  }
}

// ── Map Setup ──────────────────────────────────────────────────────────────
function initMap() {
  mapInstance = L.map('map', {
    center: [12.9716, 77.5946],
    zoom: 14,
    zoomControl: true,
    attributionControl: false,
  });

  // Dark tile layer
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    maxZoom: 19,
    opacity: 0.85,
  }).addTo(mapInstance);

  zoneLayerGroup = L.layerGroup().addTo(mapInstance);
  evacuationLayerGroup = L.layerGroup().addTo(mapInstance);
}

// ── Summary Banner ─────────────────────────────────────────────────────────
async function loadSummary() {
  const data = await apiFetch(`/api/alert/summary?disaster=${currentDisaster}`);
  if (!data) return;

  document.getElementById('stat-red').textContent   = data.red_cells;
  document.getElementById('stat-blue').textContent  = data.blue_cells;
  document.getElementById('stat-green').textContent = data.green_cells;
  document.getElementById('stat-pop').textContent   = data.at_risk_population.toLocaleString();

  const badge = document.getElementById('incident-badge');
  const level = document.getElementById('incident-level');
  level.textContent = data.alert_level;
  badge.className = 'incident-badge';
  if (data.alert_level === 'WARNING')    badge.classList.add('warning');
  if (data.alert_level === 'MONITORING') badge.classList.add('monitoring');
}

// ── Zone Map ───────────────────────────────────────────────────────────────
async function loadZones() {
  const data = await apiFetch(`/api/zones?disaster=${currentDisaster}`);
  if (!data) return;
  allZonesData = data;

  zoneLayerGroup.clearLayers();

  data.grid_cells.forEach((cell, i) => {
    const [lat, lon] = cell.centroid;
    const cfg = ZONE_COLORS[cell.zone];
    const r = cell.zone === 'RED' ? 360 : cell.zone === 'BLUE' ? 300 : 260;

    const circle = L.circle([lat, lon], {
      radius: r,
      color:       cfg.stroke,
      fillColor:   cfg.fill,
      fillOpacity: 0.55,
      weight:      2,
      className:   `zone-circle zone-${cell.zone}`,
    });

    // Glow effect via SVG filter workaround — add pulsing for RED
    circle.on('add', () => {
      const el = circle.getElement();
      if (el && cell.zone === 'RED') el.style.filter = `drop-shadow(0 0 10px ${cfg.glow})`;
      else if (el) el.style.filter = `drop-shadow(0 0 6px ${cfg.glow})`;
    });

    circle.on('click', () => {
      selectCell(cell);
    });

    circle.bindTooltip(`
      <b style="color:${cfg.stroke}">${cell.zone}</b> &nbsp;·&nbsp; ${cell.cell_id}<br/>
      Pop: <b>${cell.pop}</b> &nbsp;·&nbsp; Risk: <b>${(cell.X * 100).toFixed(0)}%</b><br/>
      Priority: <b>${(cell.rescue_priority * 100).toFixed(1)}</b>
    `, { className: 'leaflet-tooltip', sticky: true });

    zoneLayerGroup.addLayer(circle);

    // Add shelter markers on GREEN cells
    if (cell.zone === 'GREEN') {
      const shelterIcon = L.divIcon({
        html: `<div style="font-size:1.4rem;line-height:1">🏫</div>`,
        iconSize: [24, 24],
        iconAnchor: [12, 12],
        className: '',
      });
      L.marker([lat, lon], { icon: shelterIcon }).addTo(zoneLayerGroup);
    }
  });
}

// ── Evacuation Routes ──────────────────────────────────────────────────────
async function loadEvacuation() {
  const data = await apiFetch('/api/evacuation');
  if (!data) return;

  evacuationLayerGroup.clearLayers();

  // Draw evacuation paths
  data.flows.forEach((flow, i) => {
    const latlngs = flow.path.map(p => [p[0], p[1]]);
    const line = L.polyline(latlngs, {
      color: '#ff832b',
      weight: 3,
      opacity: 0.75,
      dashArray: '8 5',
    });
    line.bindTooltip(`→ ${flow.to_shelter} · ${flow.people} people · ${flow.travel_min} min`);
    evacuationLayerGroup.addLayer(line);

    // Animated arrow icon at midpoint
    const mid = latlngs[Math.floor(latlngs.length / 2)];
    const arrowIcon = L.divIcon({
      html: `<div style="color:#ff832b;font-size:1.1rem;transform:rotate(45deg)">➤</div>`,
      className: '', iconSize: [20, 20], iconAnchor: [10, 10],
    });
    L.marker(mid, { icon: arrowIcon }).addTo(evacuationLayerGroup);
  });

  // Render evacuation panel
  const evList = document.getElementById('evacuation-list');
  evList.innerHTML = data.flows.map((f, i) => `
    <div class="evac-card" style="animation-delay:${i * 0.05}s">
      <div class="evac-route">➤ ${f.from_cell.replace('_',' ')} → ${f.to_shelter}</div>
      <div class="evac-detail">
        <span>👥 ${f.people} people</span>
        <span>⏱ ${f.travel_min} min</span>
      </div>
    </div>
  `).join('');

  // Shelter status
  const shList = document.getElementById('shelter-list');
  shList.innerHTML = data.shelters.map(s => {
    const pct = Math.round(s.utilization * 100);
    const fillColor = pct >= 100 ? '#da1e28' : pct >= 80 ? '#ff832b' : '#24a148';
    return `
      <div class="shelter-card">
        <div class="shelter-name">🏫 ${s.name}</div>
        <div class="shelter-bar-track">
          <div class="shelter-bar-fill" style="width:${Math.min(pct,100)}%; background:${fillColor}"></div>
        </div>
        <div class="shelter-util">${s.assigned.toLocaleString()} / ${s.capacity.toLocaleString()} (${pct}%)</div>
      </div>
    `;
  }).join('');
}

// ── Priority Queue ─────────────────────────────────────────────────────────
async function loadPriority() {
  const data = await apiFetch('/api/priority');
  if (!data) return;

  const list = document.getElementById('priority-list');
  list.innerHTML = data.ranked.map((r, i) => `
    <div class="priority-card rank-${r.rank}" style="animation-delay:${i * 0.07}s"
         onclick="selectCellById('${r.cell_id}')">
      <div class="pcard-top">
        <span class="pcard-rank r${r.rank}">#${r.rank}</span>
        <span class="pcard-pi">PI: ${r.pi.toFixed(2)}</span>
      </div>
      <div class="pcard-meta">
        <div class="pcard-row">
          <span>Cell</span>
          <span class="pcard-val" style="font-family:'JetBrains Mono',monospace;font-size:0.68rem">${r.cell_id}</span>
        </div>
        <div class="pcard-row">
          <span>Survivors Est.</span>
          <span class="pcard-val">${r.n_est} · P(alive) ${(r.p_alive * 100).toFixed(0)}%</span>
        </div>
        <div class="pcard-row">
          <span>ETA</span>
          <span class="pcard-val">${r.eta_min} min</span>
        </div>
      </div>
      <span class="pcard-team">Assigned: ${r.team}</span>
    </div>
  `).join('');

  // Budget
  const used = data.budget_used_team_hours;
  const total = data.budget_total_team_hours;
  const pct = Math.round((used / total) * 100);
  document.getElementById('budget-fill').style.width = pct + '%';
  document.getElementById('budget-text').textContent = `${used}h / ${total}h`;
}

// ── Detections (Radar Tab) ─────────────────────────────────────────────────
async function loadDetections() {
  const detections = await apiFetch('/api/detections');
  if (!detections) return;

  const list = document.getElementById('detection-list');
  list.innerHTML = detections.map((d, i) => {
    const aliveClass = d.p_alive >= 0.7 ? 'high' : d.p_alive >= 0.4 ? 'medium' : 'low';
    const badges = d.detections.map(det => {
      const typeClass = det.type === 'thermal_hotspot' ? 'thermal'
                      : det.type === 'rppg_pulse' ? 'rppg'
                      : det.type === 'rgb_person' ? 'rgb'
                      : det.type === 'acoustic_distress' ? 'acoustic' : 'phone';
      const label = det.type === 'thermal_hotspot' ? `🌡 ${det.temp_c}°C`
                  : det.type === 'rppg_pulse' ? `💓 ${det.bpm} bpm`
                  : det.type === 'rgb_person' ? `👤 Conf ${(det.conf * 100).toFixed(0)}%`
                  : det.type === 'acoustic_distress' ? `🔊 ${det.class}`
                  : `📱 ${det.device_count} devices`;
      return `<span class="det-signal-badge ${typeClass}">${label}</span>`;
    }).join('');

    return `
      <div class="detection-card" id="det-${i}" style="animation-delay:${i * 0.06}s"
           onclick="loadRadar('${d.cell_id}', ${i})">
        <div class="det-header">
          <span class="det-cell">${d.cell_id}</span>
          <span class="det-alive ${aliveClass}">P(alive): ${(d.p_alive * 100).toFixed(0)}%</span>
        </div>
        <div class="det-signals">${badges}</div>
      </div>
    `;
  }).join('');
}

// ── Bio-Radar Chart ────────────────────────────────────────────────────────
async function loadRadar(cellId, detIdx) {
  // Highlight selected
  document.querySelectorAll('.detection-card').forEach(c => c.classList.remove('active'));
  const el = document.getElementById(`det-${detIdx}`);
  if (el) el.classList.add('active');

  document.getElementById('radar-cell-label').textContent = `📡 ${cellId}`;

  if (radarInterval) clearInterval(radarInterval);

  async function refreshRadar() {
    const data = await apiFetch(`/api/radar/heartbeat?cell_id=${cellId}`);
    if (!data) return;

    const ctx = document.getElementById('radarChart').getContext('2d');
    const labels = data.signal.map((_, i) => i);

    const badge = document.getElementById('radar-status-badge');
    badge.textContent = data.alive
      ? `ALIVE · ${data.peaks.bpm.toFixed(0)} bpm · Conf ${(data.confidence * 100).toFixed(0)}%`
      : 'NO SIGNAL';
    badge.style.background = data.alive ? 'rgba(36,161,72,0.2)' : 'rgba(218,30,40,0.2)';
    badge.style.color = data.alive ? '#6fdc8c' : '#ff8389';
    badge.style.border = data.alive ? '1px solid #198038' : '1px solid #da1e28';

    document.getElementById('radar-vitals').innerHTML = `
      <div class="vital-item"><div class="vital-label">Heart Rate</div><div class="vital-value">${data.peaks.bpm.toFixed(0)} bpm</div></div>
      <div class="vital-item"><div class="vital-label">Resp. Rate</div><div class="vital-value">${data.peaks.respiration_rpm.toFixed(0)} rpm</div></div>
      <div class="vital-item"><div class="vital-label">Confidence</div><div class="vital-value">${(data.confidence * 100).toFixed(0)}%</div></div>
    `;

    if (!radarChart) {
      radarChart = new Chart(ctx, {
        type: 'line',
        data: {
          labels,
          datasets: [{
            label: 'Bio-Radar Signal',
            data: data.signal,
            borderColor: '#08bdba',
            backgroundColor: 'rgba(8,189,186,0.08)',
            borderWidth: 1.5,
            pointRadius: 0,
            tension: 0.4,
            fill: true,
          }],
        },
        options: {
          animation: false,
          responsive: true,
          plugins: { legend: { display: false } },
          scales: {
            x: { display: false },
            y: {
              display: true,
              ticks: { color: '#4d6080', font: { size: 9 } },
              grid: { color: '#1e2d4a' },
            },
          },
        },
      });
    } else {
      radarChart.data.datasets[0].data = data.signal;
      radarChart.update('none');
    }
  }

  await refreshRadar();
  radarInterval = setInterval(refreshRadar, 2500);
}

// ── Cell Detail Panel ──────────────────────────────────────────────────────
function selectCell(cell) {
  selectedCell = cell;

  const detail = document.getElementById('cell-detail');
  const assetsHtml = (cell.assets || []).map(a => `
    <div class="asset-chip">
      <span class="asset-type">${a.type}</span>
      <span class="asset-name">${a.name}</span>
    </div>
  `).join('');

  detail.innerHTML = `
    <div class="cell-info-card">
      <span class="cell-zone-badge ${cell.zone}">${cell.zone} ZONE</span>
      <div class="cell-info-rows">
        <div class="cell-row">
          <span class="cell-label">Cell ID</span>
          <span class="cell-val" style="font-size:0.65rem">${cell.cell_id}</span>
        </div>
        <div class="cell-row">
          <span class="cell-label">Hazard Score (X)</span>
          <span class="cell-val">${(cell.X * 100).toFixed(1)}%</span>
        </div>
        <div class="cell-row">
          <span class="cell-label">Uncertainty (U)</span>
          <span class="cell-val">${(cell.U * 100).toFixed(1)}%</span>
        </div>
        <div class="cell-row">
          <span class="cell-label">Green Suitability (G)</span>
          <span class="cell-val">${(cell.G * 100).toFixed(1)}%</span>
        </div>
        <div class="cell-row">
          <span class="cell-label">Population</span>
          <span class="cell-val">${cell.pop.toLocaleString()}</span>
        </div>
        <div class="cell-row">
          <span class="cell-label">Rescue Priority</span>
          <span class="cell-val">${(cell.rescue_priority * 100).toFixed(1)}</span>
        </div>
        <div class="cell-row">
          <span class="cell-label">Reachable</span>
          <span class="cell-val">${cell.reachable ? '✅ Yes' : '❌ No'}</span>
        </div>
      </div>
      <div class="zone-reason">⚡ ${cell.zone_reason}</div>
      ${assetsHtml ? `<div class="asset-list">${assetsHtml}</div>` : ''}
    </div>
  `;

  // Gauge chart
  document.getElementById('gauge-section').style.display = 'block';
  renderGauge(cell);

  // SHAP explanation
  document.getElementById('shap-section').style.display = 'block';
  renderShap(cell);
}

function selectCellById(cellId) {
  if (!allZonesData) return;
  const cell = allZonesData.grid_cells.find(c => c.cell_id === cellId);
  if (cell) selectCell(cell);
}

function renderGauge(cell) {
  const ctx = document.getElementById('gaugeChart').getContext('2d');
  if (gaugeChart) gaugeChart.destroy();

  gaugeChart = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: ['Hazard (X)', 'Uncertainty (U)', 'Staleness', 'Coverage Gap', 'Model Diverge'],
      datasets: [{
        data: [
          cell.X,
          cell.U,
          cell.uncertainty_components?.u_stale || 0,
          cell.uncertainty_components?.u_cover || 0,
          cell.uncertainty_components?.u_model || 0,
        ],
        backgroundColor: 'rgba(218,30,40,0.15)',
        borderColor: '#ff8389',
        borderWidth: 2,
        pointBackgroundColor: '#da1e28',
        pointRadius: 4,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        r: {
          min: 0, max: 1,
          ticks: { stepSize: 0.25, color: '#4d6080', font: { size: 8 }, backdropColor: 'transparent' },
          grid: { color: '#1e2d4a' },
          angleLines: { color: '#1e2d4a' },
          pointLabels: { color: '#8d9eb8', font: { size: 9 } },
        },
      },
    },
  });
}

function renderShap(cell) {
  const container = document.getElementById('shap-bars');

  // Compute pseudo-SHAP values from cell data
  const drivers = [
    { name: 'Rainfall Intensity', val: cell.X * 0.45, pos: true },
    { name: 'Elevation (Low)', val: 1 - (cell.G * 0.5), pos: true },
    { name: 'Population Density', val: Math.min(cell.pop / 3000, 1) * 0.35, pos: true },
    { name: 'Data Staleness', val: cell.uncertainty_components?.u_stale || 0, pos: true },
    { name: 'Sensor Coverage', val: 1 - (cell.uncertainty_components?.u_cover || 0), pos: false },
    { name: 'Road Accessibility', val: cell.reachable ? 0.1 : 0.8, pos: !cell.reachable },
  ].sort((a, b) => b.val - a.val);

  container.innerHTML = drivers.map(d => `
    <div class="shap-row">
      <div class="shap-label">
        <span class="shap-name">${d.name}</span>
        <span class="shap-val">+${(d.val * 100).toFixed(0)}%</span>
      </div>
      <div class="shap-track">
        <div class="shap-fill ${d.pos ? 'pos' : 'neg'}" style="width:${(d.val * 100).toFixed(0)}%"></div>
      </div>
    </div>
  `).join('');
}

// ── Tab switching ──────────────────────────────────────────────────────────
function switchTab(tab) {
  ['priority', 'evacuation', 'radar'].forEach(t => {
    document.getElementById(`panel-${t}`).classList.toggle('hidden', t !== tab);
    document.getElementById(`tab-${t}`).classList.toggle('active', t === tab);
  });

  if (tab === 'evacuation') {
    evacuationLayerGroup.addTo(mapInstance);
  } else {
    evacuationLayerGroup.remove();
  }

  if (tab !== 'radar' && radarInterval) {
    clearInterval(radarInterval);
    radarInterval = null;
  }
}

// ── API helper ─────────────────────────────────────────────────────────────
async function apiFetch(path) {
  try {
    const res = await fetch(API + path);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  } catch (e) {
    console.warn('apiFetch error:', path, e.message);
    return null;
  }
}
