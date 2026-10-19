"use strict";
// Map projection: simple equirectangular, scaled for California's latitude.
const S = 55, KX = 0.8 * S, LON0 = -126, LAT0 = 42.2;
const project = (lon, lat) => [(lon - LON0) * KX, (LAT0 - lat) * S];
const unproject = (x, y) => [x / KX + LON0, LAT0 - y / S];

// Rough outline of California as [lon, lat] points.
const OUTLINE = [
  [-124.2, 42.0], [-120.0, 42.0], [-120.0, 39.0], [-114.63, 35.0], [-114.4, 34.1],
  [-114.5, 33.7], [-114.7, 33.4], [-114.72, 32.72], [-117.12, 32.53], [-117.28, 33.0],
  [-117.9, 33.6], [-118.4, 33.75], [-118.5, 34.03], [-119.2, 34.15], [-120.45, 34.45],
  [-120.65, 35.15], [-121.3, 35.65], [-121.9, 36.3], [-121.9, 36.6], [-122.4, 37.2],
  [-122.5, 37.8], [-123.0, 38.0], [-123.7, 38.9], [-123.8, 39.7], [-124.35, 40.4],
  [-124.1, 41.0], [-124.2, 41.8],
];
const CITIES = [
  ["Los Angeles", 34.05, -118.24, "start"], ["San Francisco", 37.77, -122.42, "end"],
  ["San Diego", 32.72, -117.16, "start"], ["Sacramento", 38.58, -121.49, "start"],
  ["Fresno", 36.74, -119.79, "start"],
];
const FIELDS = {
  MedInc: ["Neighborhood income", 0.1, v => "$" + Math.round(v * 1e4).toLocaleString("en-US")],
  HouseAge: ["House age", 1, v => Math.round(v) + " years"],
  AveRooms: ["Rooms per home", 0.1, v => v.toFixed(1)],
  AveBedrms: ["Bedrooms per home", 0.05, v => v.toFixed(2)],
  AveOccup: ["People per home", 0.1, v => v.toFixed(1)],
  Population: ["Neighborhood population", 10, v => Math.round(v).toLocaleString("en-US")],
};

const $ = id => document.getElementById(id);
const svg = $("map"), values = {};
const money = (n, step = 1000) => "$" + (Math.round(n / step) * step).toLocaleString("en-US");

function inside(lon, lat) {
  let c = false;
  for (let i = 0, j = OUTLINE.length - 1; i < OUTLINE.length; j = i++) {
    const [xi, yi] = OUTLINE[i], [xj, yj] = OUTLINE[j];
    if ((yi > lat) !== (yj > lat) && lon < ((xj - xi) * (lat - yi)) / (yj - yi) + xi) c = !c;
  }
  return c;
}

function setPin(lat, lon) {
  if (!inside(lon, lat)) return;
  values.Latitude = lat; values.Longitude = lon;
  const [x, y] = project(lon, lat);
  $("pin").style.transform = `translate(${x}px, ${y}px)`;
  schedule();
}

function drawMap() {
  $("state").setAttribute("d", "M" + OUTLINE.map(([lo, la]) => project(lo, la).join(",")).join("L") + "Z");
  for (const [name, lat, lon, anchor] of CITIES) {
    const [x, y] = project(lon, lat);
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.innerHTML = `<circle cx="${x}" cy="${y}" r="3.5"></circle>
      <text x="${x + (anchor === "end" ? -8 : 8)}" y="${y + 4}" text-anchor="${anchor}">${name}</text>`;
    g.addEventListener("click", e => { e.stopPropagation(); setPin(lat, lon); });
    $("cities").append(g);
  }
  svg.addEventListener("click", e => {
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(svg.getScreenCTM().inverse());
    const [lon, lat] = unproject(p.x, p.y);
    setPin(lat, lon);
  });
  svg.addEventListener("keydown", e => {
    const d = { ArrowUp: [0.15, 0], ArrowDown: [-0.15, 0], ArrowLeft: [0, -0.15], ArrowRight: [0, 0.15] }[e.key];
    if (!d) return;
    e.preventDefault();
    setPin(values.Latitude + d[0], values.Longitude + d[1]);
  });
}

function drawControls(meta) {
  for (const [key, [label, step, fmt]] of Object.entries(FIELDS)) {
    const m = meta.features[key];
    values[key] = m.default;
    const field = document.createElement("div");
    field.className = "field";
    field.innerHTML = `<div class="field-head"><label for="f-${key}">${label}</label><output for="f-${key}"></output></div>
      <input type="range" id="f-${key}" min="${m.min}" max="${m.max}" step="${step}" value="${m.default}">`;
    const input = field.querySelector("input"), out = field.querySelector("output");
    const sync = () => { values[key] = +input.value; out.textContent = fmt(+input.value); };
    input.addEventListener("input", () => { sync(); schedule(); });
    sync();
    $("controls").append(field);
  }
}

let timer, model;
const schedule = () => { clearTimeout(timer); timer = setTimeout(predict, 100); };

// Walk one decision tree exported from scikit-learn.
function treeValue(t, x) {
  let n = 0;
  while (t.l[n] !== -1) n = x[t.f[n]] <= t.t[n] ? t.l[n] : t.r[n];
  return t.v[n];
}

function predict() {
  if (!model) return;
  const x = model.features.map(f => values[f]);
  const sum = model.trees.reduce((s, t) => s + treeValue(t, x), 0);
  $("price").textContent = money((model.init + model.lr * sum) * 1e5);
  $("margin").textContent = `Typically within about ${money(model.meta.mae)} of the real value.`;
}

async function init() {
  drawMap();
  try {
    const res = await fetch("model.json");
    if (!res.ok) throw new Error("model.json not found. Run python train.py, then commit docs/model.json.");
    model = await res.json();
    drawControls(model.meta);
    const { Latitude: la, Longitude: lo } = model.meta.features;
    setPin(la.default, lo.default);
  } catch (err) {
    $("price").textContent = "No estimate";
    $("margin").textContent = err instanceof TypeError
      ? "Could not load model.json. Open the page through a web server (python -m http.server), not by double-clicking the file."
      : err.message;
  }
}
init();
