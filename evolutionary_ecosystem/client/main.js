/**
 * Evolutionary Ecosystem — Three.js client with procedural creature meshes.
 *
 * Based on creature_ecosystem client with additions:
 * - Epoch/weather display
 * - Food items rendered as green spheres
 * - Energy/HP bars on creature cards
 * - Population statistics panel
 * - Population behavior profile bar chart
 */

import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js";

// ---------------------------------------------------------------------------
// Scene setup
// ---------------------------------------------------------------------------

const container = document.getElementById("canvas-container");
const W = () => container.clientWidth;
const H = () => container.clientHeight;

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.shadowMap.enabled = true;
renderer.setSize(W(), H());
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87ceeb);

const camera = new THREE.PerspectiveCamera(55, W() / H(), 0.1, 200);
camera.position.set(10, 7, 20);
camera.lookAt(10, 0, 7);

const ambient = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambient);
const sun = new THREE.DirectionalLight(0xfff8e1, 1.2);
sun.position.set(15, 20, 10);
sun.castShadow = true;
scene.add(sun);

const groundGeo = new THREE.PlaneGeometry(20, 14);
const groundMat = new THREE.MeshLambertMaterial({ color: 0x7ec850 });
const ground = new THREE.Mesh(groundGeo, groundMat);
ground.rotation.x = -Math.PI / 2;
ground.position.set(10, 0, 7);
ground.receiveShadow = true;
scene.add(ground);

const gridHelper = new THREE.GridHelper(20, 20, 0x4a9c2d, 0x4a9c2d);
gridHelper.position.set(10, 0.01, 7);
scene.add(gridHelper);

window.addEventListener("resize", () => {
  camera.aspect = W() / H();
  camera.updateProjectionMatrix();
  renderer.setSize(W(), H());
});

// ---------------------------------------------------------------------------
// Procedural mesh builders
// ---------------------------------------------------------------------------

function buildCreatureMesh(ap) {
  const group = new THREE.Group();

  const primaryColor   = new THREE.Color().setHSL(ap.primary_hue   / 360, 0.70, 0.55);
  const secondaryColor = new THREE.Color().setHSL(ap.secondary_hue / 360, 0.65, 0.48);
  const tailColor      = new THREE.Color().setHSL((ap.tail_hue ?? 90)  / 360, 0.65, 0.50);
  const limbColor      = new THREE.Color().setHSL((ap.limb_hue ?? 270) / 360, 0.65, 0.50);
  const eyeColor       = new THREE.Color(0x111111);

  const mat1    = new THREE.MeshStandardMaterial({ color: primaryColor,   roughness: 0.7 });
  const mat2    = new THREE.MeshStandardMaterial({ color: secondaryColor, roughness: 0.6 });
  const matTail = new THREE.MeshStandardMaterial({ color: tailColor,      roughness: 0.65 });
  const matLimb = new THREE.MeshStandardMaterial({ color: limbColor,      roughness: 0.7 });
  const matEye  = new THREE.MeshStandardMaterial({ color: eyeColor });

  const r = ap.body_radius;

  const bodyCore = new THREE.Group();
  bodyCore.position.y = r;
  group.add(bodyCore);

  // Body: slightly squashed sphere (wider than tall, slightly elongated front-back)
  const body = new THREE.Mesh(new THREE.SphereGeometry(r, 16, 12), mat1);
  body.scale.set(1.0, 0.85, 1.15);  // wider front-back, slightly flat on top
  body.castShadow = true;
  bodyCore.add(body);

  // Head: distinct sphere sitting on top-front of body
  const headR = r * ap.head_scale * 0.7;
  const head = new THREE.Mesh(new THREE.SphereGeometry(headR, 16, 12), mat2);
  head.position.set(0, r * 0.65 + headR * 0.3, r * 0.5 + headR * 0.2);
  head.castShadow = true;
  bodyCore.add(head);

  const eyeR = r * 0.14;
  for (const side of [-1, 1]) {
    const eye = new THREE.Mesh(new THREE.SphereGeometry(eyeR, 8, 8), matEye);
    eye.position.set(side * headR * 0.52, head.position.y + headR * 0.1, head.position.z + headR * 0.82);
    bodyCore.add(eye);
    const shine = new THREE.Mesh(
      new THREE.SphereGeometry(eyeR * 0.4, 6, 6),
      new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff }),
    );
    shine.position.set(eye.position.x + eyeR * 0.3, eye.position.y + eyeR * 0.3, eye.position.z + eyeR * 0.6);
    bodyCore.add(shine);
  }

  const earGeo = ap.ear_pointy > 0.5
    ? new THREE.ConeGeometry(r * 0.18, r * ap.ear_pointy * 0.7 + 0.05, 8)
    : new THREE.SphereGeometry(r * 0.22, 8, 8);
  for (const side of [-1, 1]) {
    const ear = new THREE.Mesh(earGeo, mat2);
    ear.position.set(side * headR * 0.58, head.position.y + headR * 0.85, head.position.z - headR * 0.1);
    if (ap.ear_pointy > 0.5) ear.rotation.z = -side * 0.2;
    bodyCore.add(ear);
  }

  const nose = new THREE.Mesh(
    new THREE.SphereGeometry(r * 0.09, 8, 8),
    new THREE.MeshStandardMaterial({ color: 0x3a1a0a }),
  );
  nose.position.set(0, head.position.y - headR * 0.1, head.position.z + headR * 0.95);
  bodyCore.add(nose);

  const tailLen = ap.tail_length;
  const tail = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.07, r * 0.04, tailLen, 8), matTail);
  tail.position.set(0, 0, -r * 0.9 - tailLen / 2);
  tail.rotation.x = Math.PI / 4;
  bodyCore.add(tail);

  const limbGeo = new THREE.CylinderGeometry(r * 0.11, r * 0.09, ap.limb_length, 8);
  const limbOffsets = [[-0.55*r, -0.55*r], [-0.55*r, 0.55*r], [0.55*r, -0.55*r], [0.55*r, 0.55*r]];
  const legs = [];
  for (const [ox, oz] of limbOffsets) {
    const limb = new THREE.Mesh(limbGeo, matLimb);
    limb.position.set(ox, ap.limb_length * 0.5, oz + r * 0.2);
    group.add(limb);
    legs.push(limb);
    const paw = new THREE.Mesh(new THREE.SphereGeometry(r * 0.14, 8, 8), matLimb);
    paw.position.set(ox, 0.05, oz + r * 0.2);
    group.add(paw);
  }

  group.userData.ap = ap;
  group.userData.bodyCore = bodyCore;
  group.userData.legs = legs;
  return group;
}

function buildFlowerMesh() {
  const g = new THREE.Group();
  const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.03, 0.35, 6), new THREE.MeshStandardMaterial({ color: 0x2e8b22 }));
  stem.position.y = 0.175;
  g.add(stem);
  // Petals — ring of small spheres around center
  const petalMat = new THREE.MeshStandardMaterial({ color: 0xff69b4, emissive: 0xdd2266, emissiveIntensity: 0.15 });
  const center = new THREE.Mesh(new THREE.SphereGeometry(0.06, 8, 8), new THREE.MeshStandardMaterial({ color: 0xffdd00 }));
  center.position.y = 0.4;
  g.add(center);
  for (let i = 0; i < 5; i++) {
    const angle = (i / 5) * Math.PI * 2;
    const petal = new THREE.Mesh(new THREE.SphereGeometry(0.07, 6, 6), petalMat);
    petal.scale.set(1, 0.6, 1);
    petal.position.set(Math.cos(angle) * 0.09, 0.4, Math.sin(angle) * 0.09);
    g.add(petal);
  }
  // Leaf
  const leaf = new THREE.Mesh(new THREE.SphereGeometry(0.05, 6, 4), new THREE.MeshStandardMaterial({ color: 0x3aaa22 }));
  leaf.scale.set(1.5, 0.4, 1);
  leaf.position.set(0.06, 0.2, 0);
  g.add(leaf);
  return g;
}

function buildTreeMesh() {
  const g = new THREE.Group();
  const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.1, 0.6, 8), new THREE.MeshStandardMaterial({ color: 0x8b5a2b, roughness: 0.9 }));
  trunk.position.y = 0.3;
  g.add(trunk);
  // Layered canopy
  const leafMat = new THREE.MeshStandardMaterial({ color: 0x2d8b22, roughness: 0.8 });
  const canopy1 = new THREE.Mesh(new THREE.SphereGeometry(0.42, 10, 8), leafMat);
  canopy1.position.y = 0.85;
  canopy1.scale.set(1, 0.8, 1);
  g.add(canopy1);
  const canopy2 = new THREE.Mesh(new THREE.SphereGeometry(0.3, 8, 6), leafMat);
  canopy2.position.y = 1.1;
  g.add(canopy2);
  g.castShadow = true;
  return g;
}

function buildRockMesh() {
  const g = new THREE.Group();
  const rockMat = new THREE.MeshStandardMaterial({ color: 0x888888, roughness: 0.95 });
  // Main rock
  const geo = new THREE.SphereGeometry(0.2, 7, 5);
  geo.applyMatrix4(new THREE.Matrix4().makeScale(1.1, 0.7, 0.95));
  const main = new THREE.Mesh(geo, rockMat);
  main.position.y = 0.14;
  main.rotation.y = 0.4;
  g.add(main);
  // Smaller rock beside it
  const geo2 = new THREE.SphereGeometry(0.1, 6, 4);
  geo2.applyMatrix4(new THREE.Matrix4().makeScale(1.0, 0.65, 0.9));
  const small = new THREE.Mesh(geo2, new THREE.MeshStandardMaterial({ color: 0x777777, roughness: 0.95 }));
  small.position.set(0.15, 0.07, 0.08);
  g.add(small);
  return g;
}

function buildFoodMesh() {
  // Static food item — golden apple/berry shape
  const g = new THREE.Group();
  const body = new THREE.Mesh(new THREE.SphereGeometry(0.12, 10, 8), new THREE.MeshStandardMaterial({ color: 0xffd700, emissive: 0xffaa00, emissiveIntensity: 0.3 }));
  body.position.y = 0.18;
  body.scale.set(1, 0.85, 1);
  g.add(body);
  // Little stem
  const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.015, 0.015, 0.08, 4), new THREE.MeshStandardMaterial({ color: 0x5a3a1a }));
  stem.position.y = 0.3;
  g.add(stem);
  // Leaf on stem
  const leaf = new THREE.Mesh(new THREE.SphereGeometry(0.03, 4, 4), new THREE.MeshStandardMaterial({ color: 0x44aa22 }));
  leaf.scale.set(1.5, 0.5, 1);
  leaf.position.set(0.03, 0.3, 0);
  g.add(leaf);
  g.castShadow = true;
  return g;
}

function buildBallMesh() {
  const g = new THREE.Group();
  const ball = new THREE.Mesh(new THREE.SphereGeometry(0.16, 14, 10), new THREE.MeshStandardMaterial({ color: 0xff4466, roughness: 0.3, metalness: 0.2 }));
  ball.position.y = 0.16;
  ball.castShadow = true;
  g.add(ball);
  // Stripe
  const stripe = new THREE.Mesh(
    new THREE.TorusGeometry(0.155, 0.02, 6, 20),
    new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.3 })
  );
  stripe.position.y = 0.16;
  stripe.rotation.x = Math.PI / 2;
  g.add(stripe);
  return g;
}

function buildPredatorMesh() {
  const g = new THREE.Group();
  const matBody = new THREE.MeshStandardMaterial({ color: 0x1a0a0a, roughness: 0.4, metalness: 0.3 });
  const matAccent = new THREE.MeshStandardMaterial({ color: 0x550000, emissive: 0x330000, emissiveIntensity: 0.3 });
  const matEye = new THREE.MeshStandardMaterial({ color: 0xff0000, emissive: 0xff2200, emissiveIntensity: 1.0 });

  // Main body — low slung, wider and flatter than creatures
  const body = new THREE.Mesh(new THREE.SphereGeometry(0.4, 12, 10), matBody);
  body.scale.set(1.3, 0.7, 1.6);  // wide, flat, long
  body.position.y = 0.35;
  body.castShadow = true;
  g.add(body);

  // Head — angular, wedge-shaped using a cone
  const head = new THREE.Mesh(new THREE.ConeGeometry(0.25, 0.45, 6), matBody);
  head.rotation.x = -Math.PI / 2;  // point forward
  head.position.set(0, 0.4, 0.65);
  head.castShadow = true;
  g.add(head);

  // Jaw / snout
  const jaw = new THREE.Mesh(new THREE.ConeGeometry(0.15, 0.3, 5), matAccent);
  jaw.rotation.x = -Math.PI / 2;
  jaw.position.set(0, 0.3, 0.75);
  g.add(jaw);

  // Eyes — glowing red, menacing
  for (const side of [-1, 1]) {
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.07, 8, 8), matEye);
    eye.position.set(side * 0.15, 0.48, 0.6);
    g.add(eye);
    // Eye glow
    const glow = new THREE.Mesh(new THREE.SphereGeometry(0.04, 6, 6),
      new THREE.MeshStandardMaterial({ color: 0xffff00, emissive: 0xffaa00, emissiveIntensity: 1.5 }));
    glow.position.set(side * 0.15, 0.49, 0.63);
    g.add(glow);
  }

  // Spines along the back
  for (let i = 0; i < 5; i++) {
    const spine = new THREE.Mesh(new THREE.ConeGeometry(0.04, 0.15 + i * 0.02, 4), matAccent);
    spine.position.set(0, 0.6 - i * 0.02, -0.2 + i * 0.18);
    g.add(spine);
  }

  // Legs — four stocky limbs
  const legMat = new THREE.MeshStandardMaterial({ color: 0x1a0808, roughness: 0.6 });
  const legOffsets = [[-0.25, -0.35], [-0.25, 0.25], [0.25, -0.35], [0.25, 0.25]];
  for (const [ox, oz] of legOffsets) {
    const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.05, 0.3, 6), legMat);
    leg.position.set(ox, 0.12, oz);
    g.add(leg);
    // Claws
    const claw = new THREE.Mesh(new THREE.ConeGeometry(0.04, 0.08, 4), matAccent);
    claw.rotation.x = Math.PI / 2;
    claw.position.set(ox, 0.03, oz + 0.05);
    g.add(claw);
  }

  // Tail — thick and spiked
  const tail = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.03, 0.5, 6), matBody);
  tail.position.set(0, 0.35, -0.7);
  tail.rotation.x = -Math.PI / 6;
  g.add(tail);

  return g;
}

// Food dots for ecosystem food
function buildFoodDot() {
  const g = new THREE.Group();
  // Bright green berry — large enough to be visible from above
  const berry = new THREE.Mesh(
    new THREE.SphereGeometry(0.12, 8, 6),
    new THREE.MeshStandardMaterial({ color: 0x44ff44, emissive: 0x22ee22, emissiveIntensity: 0.5 }),
  );
  berry.position.y = 0.15;
  berry.scale.set(1, 0.8, 1);
  berry.castShadow = true;
  g.add(berry);
  // Leaf
  const leaf = new THREE.Mesh(
    new THREE.SphereGeometry(0.04, 4, 4),
    new THREE.MeshStandardMaterial({ color: 0x228822 }),
  );
  leaf.scale.set(1.8, 0.4, 1);
  leaf.position.set(0.04, 0.24, 0);
  g.add(leaf);
  return g;
}

const ITEM_BUILDERS = { flower: buildFlowerMesh, tree: buildTreeMesh, rock: buildRockMesh, food: buildFoodMesh, ball: buildBallMesh };

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const creatureMeshes = {};  // id → THREE.Group
const itemMeshes = {};
const foodDotMeshes = [];   // array of meshes for ecosystem food
let predatorMesh = null;
let selectedId = null;
let ws = null;
let namesVisible = false;
let bearEnabled = true;
const thoughtBubbles = {};
const nameLabels = {};
const convLog = [];

const MOOD_EMOJI = { happy: "😊", playful: "😜", neutral: "😐", cautious: "😟", excited: "🤩", sleepy: "😴", annoyed: "😤" };
const WEATHER_EMOJI = { mild: "☀️", storm: "⛈️", heat: "🔥", cold: "❄️" };
let currentWeather = "mild";
let weatherParticles = [];

const WEATHER_SETTINGS = {
  mild:  { sky: 0x87ceeb, ambient: 0xffffff, ambientI: 0.6, sunI: 1.2, fog: null,       ground: 0x7ec850 },
  storm: { sky: 0x3a3a50, ambient: 0x8888aa, ambientI: 0.3, sunI: 0.4, fog: 0x3a3a50,   ground: 0x4a7a3a },
  heat:  { sky: 0xffcc77, ambient: 0xffddaa, ambientI: 0.7, sunI: 1.5, fog: 0xffeecc,    ground: 0xa89840 },
  cold:  { sky: 0xc8d8f0, ambient: 0xaabbdd, ambientI: 0.5, sunI: 0.7, fog: 0xd0e0f0,   ground: 0xd0dde8 },
};

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------

function connect() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.onopen = () => {
    document.getElementById("status").textContent = "Connected";
    document.getElementById("status").className = "status connected";
  };
  ws.onclose = () => {
    document.getElementById("status").textContent = "Disconnected";
    document.getElementById("status").className = "status disconnected";
    setTimeout(connect, 2000);
  };
  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data);
      if (msg.type === "state") handleState(msg);
      else if (msg.type === "birth") handleBirth(msg);
      else if (msg.type === "warning") addConvEntry(msg.text, null, null, "warning");
    } catch (e) { console.error("WS parse error", e); }
  };
}

function send(obj) {
  if (ws && ws.readyState === 1) {
    ws.send(JSON.stringify(obj));
  } else {
    console.warn("WebSocket not ready, state:", ws?.readyState, "msg:", obj);
  }
}

// ---------------------------------------------------------------------------
// State handler
// ---------------------------------------------------------------------------

function handleState(state) {
  const creatures = state.creatures || [];
  const items = state.items || [];
  const food = state.food || [];
  const predator = state.predator;

  document.getElementById("hud-count").textContent = creatures.length;
  document.getElementById("hud-paused").textContent = state.paused ? "Paused" : "Running";
  document.getElementById("hud-paused").className = state.paused ? "status disconnected" : "status running";

  // Epoch / weather
  if (state.epoch) {
    document.getElementById("epoch-name").textContent = state.epoch;
    const weatherEl = document.getElementById("weather-icon");
    weatherEl.textContent = `${WEATHER_EMOJI[state.weather] || "☀️"} ${state.weather}`;
    weatherEl.className = `weather-tag weather-${state.weather}`;
    applyWeather(state.weather);
  }

  // Sync breeding mode dropdowns
  if (state.recombination) {
    const sel = document.getElementById("sel-recombination");
    if (sel && sel.value !== state.recombination) sel.value = state.recombination;
  }
  if (state.ploidy) {
    const sel = document.getElementById("sel-ploidy");
    if (sel && sel.value !== state.ploidy) sel.value = state.ploidy;
  }

  // Pop stats
  if (state.total_births !== undefined) {
    document.getElementById("pop-births").textContent = state.total_births;
    document.getElementById("pop-deaths").textContent = state.total_deaths;
    document.getElementById("pop-tick").textContent = state.tick || 0;
  }

  // BEAR state
  const bearOff = state.bear_disabled;
  document.getElementById("hud-bear").textContent = bearOff ? "BEAR: INACTIVE" : "BEAR: ACTIVE";
  document.getElementById("hud-bear").className = bearOff ? "bear-badge bear-off-badge" : "bear-badge bear-on-badge";
  document.body.classList.toggle("bear-disabled", bearOff);
  bearEnabled = !bearOff;

  // Sync creature meshes
  const aliveIds = new Set(creatures.map(c => c.id));
  for (const id of Object.keys(creatureMeshes)) {
    if (!aliveIds.has(id)) {
      scene.remove(creatureMeshes[id]);
      delete creatureMeshes[id];
      if (nameLabels[id]) { nameLabels[id].remove(); delete nameLabels[id]; }
      if (thoughtBubbles[id]) { thoughtBubbles[id].el.remove(); delete thoughtBubbles[id]; }
    }
  }

  for (const c of creatures) {
    if (!creatureMeshes[c.id]) {
      const mesh = buildCreatureMesh(c.appearance);
      scene.add(mesh);
      creatureMeshes[c.id] = mesh;
    }
    const mesh = creatureMeshes[c.id];
    mesh.position.set(c.x, 0, c.y);
    mesh.rotation.y = -c.heading + Math.PI / 2;

    // Animations
    const t = performance.now() / 1000;
    const bc = mesh.userData.bodyCore;
    if (bc) {
      if (c.animation === "roll") {
        bc.rotation.z = t * 4;
      } else if (c.animation === "bounce" || c.skills?.bounce_gait) {
        mesh.position.y = Math.abs(Math.sin(t * 3 + c.x)) * 0.15;
        bc.rotation.z = 0;
      } else if (c.animation === "sneak") {
        mesh.scale.y = 0.7;
        bc.rotation.z = 0;
      } else {
        mesh.scale.y = 1;
        bc.rotation.z = 0;
        mesh.position.y = 0;
      }
    }

    // Fading
    mesh.traverse(child => { if (child.material) child.material.opacity = c.fading ? 0.5 : 1; });

    // Thought bubble
    updateThought(c);
  }

  // Clean up stale item meshes
  const aliveItemIds = new Set(items.map(i => i.id));
  for (const id of Object.keys(itemMeshes)) {
    if (!aliveItemIds.has(id)) {
      scene.remove(itemMeshes[id]);
      delete itemMeshes[id];
    }
  }

  // Items
  for (const item of items) {
    if (!itemMeshes[item.id]) {
      const builder = ITEM_BUILDERS[item.type];
      if (!builder) continue;
      const mesh = builder();
      scene.add(mesh);
      itemMeshes[item.id] = mesh;
    }
    const mesh = itemMeshes[item.id];
    mesh.position.set(item.x, 0, item.y);
    mesh.visible = item.active;
  }

  // Ecosystem food dots
  while (foodDotMeshes.length < food.length) {
    const dot = buildFoodDot();
    scene.add(dot);
    foodDotMeshes.push(dot);
  }
  for (let i = 0; i < foodDotMeshes.length; i++) {
    if (i < food.length) {
      foodDotMeshes[i].visible = true;
      foodDotMeshes[i].position.set(food[i].x, 0, food[i].y);
    } else {
      foodDotMeshes[i].visible = false;
    }
  }

  // Predator
  if (predator && predator.active) {
    if (!predatorMesh) {
      predatorMesh = buildPredatorMesh();
      scene.add(predatorMesh);
    }
    predatorMesh.visible = true;
    predatorMesh.position.set(predator.x, 0, predator.y);
    predatorMesh.rotation.y = -predator.heading + Math.PI / 2;
  } else if (predatorMesh) {
    predatorMesh.visible = false;
  }

  // Update sidebar
  updateCreatureCards(creatures);

  // Update behavior panels
  updatePopBehavior(creatures);
  updateDivergence(creatures);

  // Selected creature trace
  if (selectedId) {
    const sel = creatures.find(c => c.id === selectedId);
    if (sel) {
      updateBearTrace(sel);
      updateBehaviorProfile(sel);
      updateFamilyTree(sel);
    }
  }

  // Avg generation
  if (creatures.length > 0) {
    const avgGen = creatures.reduce((s, c) => s + c.generation, 0) / creatures.length;
    document.getElementById("pop-avg-gen").textContent = avgGen.toFixed(1);
  }
}

function handleBirth(msg) {
  const c = msg.creature;
  addConvEntry(`${c.name} was born! (Gen ${c.generation})`, null, null, "birth");
}

// ---------------------------------------------------------------------------
// Thought bubbles
// ---------------------------------------------------------------------------

const overlay = document.getElementById("thought-overlay");

function updateThought(c) {
  if (c.thought && c.thought.length > 0) {
    if (!thoughtBubbles[c.id]) {
      const el = document.createElement("div");
      el.className = "thought-bubble";
      overlay.appendChild(el);
      thoughtBubbles[c.id] = { el, text: "", timer: 0 };
    }
    const tb = thoughtBubbles[c.id];
    if (tb.text !== c.thought) {
      tb.text = c.thought;
      tb.el.textContent = c.thought;
      tb.timer = performance.now() + 4000;
      if (c.speak_to) {
        tb.el.classList.add("directed");
      } else {
        tb.el.classList.remove("directed");
      }

      // Log to conversation
      const speakerName = c.name;
      const hue = c.appearance?.primary_hue || 0;
      addConvEntry(c.thought, speakerName, `hsl(${hue}, 70%, 55%)`, c.speak_to ? "directed" : "");
    }
  }

  if (thoughtBubbles[c.id]) {
    const tb = thoughtBubbles[c.id];
    if (performance.now() > tb.timer) {
      tb.el.remove();
      delete thoughtBubbles[c.id];
      return;
    }
    // Position bubble above creature in screen space
    const pos = new THREE.Vector3(c.x, 1.2, c.y);
    pos.project(camera);
    const x = (pos.x * 0.5 + 0.5) * W();
    const y = (-pos.y * 0.5 + 0.5) * H();
    tb.el.style.left = `${x}px`;
    tb.el.style.top = `${y - 40}px`;
    tb.el.style.transform = "translate(-50%, -100%)";
  }
}

// ---------------------------------------------------------------------------
// Creature cards
// ---------------------------------------------------------------------------

function updateCreatureCards(creatures) {
  const list = document.getElementById("creature-list");
  const existingCards = {};
  for (const card of list.children) {
    existingCards[card.dataset.id] = card;
  }

  for (const c of creatures) {
    let card = existingCards[c.id];
    if (!card) {
      card = document.createElement("div");
      card.className = "creature-card";
      card.dataset.id = c.id;
      card.addEventListener("click", () => selectCreature(c.id));
      list.appendChild(card);
    }

    const hue = c.appearance?.primary_hue || 0;
    const emoji = MOOD_EMOJI[c.mood] || "😐";
    const breedClass = c.breed_ready ? "breed-ready" : "";
    const rabidClass = c.is_rabid ? "rabid-creature" : "";
    const selectedClass = c.id === selectedId ? "selected" : "";

    card.className = `creature-card ${breedClass} ${rabidClass} ${selectedClass}`.trim();

    const hpPct = Math.max(0, Math.min(100, c.hp));
    const energyPct = Math.max(0, Math.min(100, c.energy));
    const happyPct = Math.max(0, Math.min(100, c.happiness));

    card.innerHTML = `
      <div class="card-header">
        <span class="color-swatch" style="background:hsl(${hue},70%,55%)"></span>
        <strong>${c.name}</strong>
        <span class="mood-emoji">${emoji}</span><span class="happy-num">${Math.round(happyPct)}</span>
        ${c.generation > 0 ? `<span class="gen-badge">G${c.generation}</span>` : ""}
        ${c.is_rabid ? `<span class="rabid-badge">RABID</span>` : ""}
      </div>
      <div class="happiness-bar"><div class="happiness-fill" style="width:${happyPct}%"></div></div>
      <div class="stat-bars">
        <div class="mini-bar"><span class="mini-label">HP</span><div class="mini-bar-bg"><div class="mini-bar-fill hp-fill" style="width:${hpPct}%"></div></div><span class="bar-num">${Math.round(hpPct)}</span></div>
        <div class="mini-bar"><span class="mini-label">EN</span><div class="mini-bar-bg"><div class="mini-bar-fill energy-fill" style="width:${energyPct}%"></div></div><span class="bar-num">${Math.round(energyPct)}</span></div>
      </div>
      <div class="card-meta">${c.intent || "idle"} · Age ${c.age?.toFixed(0) || 0}s · K:${c.kills || 0}</div>
    `;

    delete existingCards[c.id];
  }

  // Remove cards for dead creatures
  for (const id of Object.keys(existingCards)) {
    existingCards[id].remove();
  }
}

function selectCreature(id) {
  if (selectedId === id) {
    selectedId = null;
    document.getElementById("behavior-section").style.display = "none";
    document.getElementById("family-section").style.display = "none";
    document.getElementById("bear-trace").innerHTML = '<div class="trace-empty">No creature selected</div>';
  } else {
    selectedId = id;
  }
}

// ---------------------------------------------------------------------------
// BEAR trace panel
// ---------------------------------------------------------------------------

function updateBearTrace(c) {
  const el = document.getElementById("bear-trace");
  const r = c.last_retrieval;
  if (!r) {
    el.innerHTML = '<div class="trace-empty">No retrieval yet</div>';
    return;
  }

  const scoreW = Math.round(Math.min(r.score, 1) * 100);
  const bearOff = r.text?.includes("BEAR disabled");
  const traceClass = bearOff ? "trace-block bear-off-trace" : "trace-block";
  const textClass = bearOff ? "trace-instruction bear-off-text" : "trace-instruction";

  el.innerHTML = `
    <div class="${traceClass}">
      <div class="trace-trigger">Trigger: ${r.trigger}</div>
      <div class="trace-query">"${r.query}"</div>
      <div class="${textClass}">${r.text}</div>
      <div class="trace-score">
        Score: ${r.score.toFixed(3)}
        <span class="trace-score-bar" style="width:${scoreW}px"></span>
      </div>
    </div>
  `;
}

// ---------------------------------------------------------------------------
// Behavior profile
// ---------------------------------------------------------------------------

function updateBehaviorProfile(c) {
  const section = document.getElementById("behavior-section");
  const el = document.getElementById("behavior-profile");

  if (!c.behavior) {
    section.style.display = "none";
    return;
  }

  section.style.display = "";
  const GENE_COLORS = {
    personality: "#e94560", social_style: "#4ecdc4", reaction_pattern: "#ffe66d",
    foraging: "#66bb6a", predator_defense: "#ff7043", climate_survival: "#42a5f5",
    territorial: "#ab47bc", movement_style: "#ffa726", none: "#888",
  };

  let html = "";
  for (const [name, data] of Object.entries(c.behavior)) {
    const pct = Math.round(data.strength * 100);
    const color = GENE_COLORS[data.gene_category] || "#888";
    html += `
      <div class="pop-row">
        <span class="pop-label">${name}</span>
        <div class="pop-bar-bg"><div class="pop-bar-fill" style="width:${pct}%;background:${color}"></div></div>
        <span class="pop-pct">${pct}%</span>
      </div>
    `;
  }
  el.innerHTML = html;
}

// ---------------------------------------------------------------------------
// Family tree
// ---------------------------------------------------------------------------

function updateFamilyTree(c) {
  const section = document.getElementById("family-section");
  const el = document.getElementById("family-tree");

  if (!c.parents) {
    section.style.display = "none";
    return;
  }

  section.style.display = "";
  const hue = c.appearance?.primary_hue || 0;
  el.innerHTML = `
    <div class="ftree-child">
      <div class="ftree-node">
        <span class="ftree-dot" style="background:hsl(${hue},70%,55%)"></span>
        <strong>${c.name}</strong> <span class="gen-badge">G${c.generation}</span>
      </div>
    </div>
    <div class="ftree-line"></div>
    <div class="ftree-parents ftree-parent-row">
      <div class="ftree-parent"><span class="ftree-label">Parent A</span> ${c.parents[0]}</div>
      <div class="ftree-parent"><span class="ftree-label">Parent B</span> ${c.parents[1]}</div>
    </div>
  `;
}

// ---------------------------------------------------------------------------
// Population behavior
// ---------------------------------------------------------------------------

function updatePopBehavior(creatures) {
  const el = document.getElementById("pop-behavior-panel");
  const withBehavior = creatures.filter(c => c.behavior);
  if (withBehavior.length === 0) return;

  const situations = Object.keys(withBehavior[0].behavior);
  const GENE_COLORS = {
    personality: "#e94560", social_style: "#4ecdc4", reaction_pattern: "#ffe66d",
    foraging: "#66bb6a", predator_defense: "#ff7043", climate_survival: "#42a5f5",
    territorial: "#ab47bc", none: "#888",
  };

  let html = "";
  for (const sit of situations) {
    let total = 0;
    const catCounts = {};
    for (const c of withBehavior) {
      const data = c.behavior[sit];
      if (data) {
        total += data.strength;
        catCounts[data.gene_category] = (catCounts[data.gene_category] || 0) + 1;
      }
    }
    const avg = total / withBehavior.length;
    const pct = Math.round(avg * 100);
    const dominant = Object.entries(catCounts).sort((a, b) => b[1] - a[1])[0];
    const color = dominant ? (GENE_COLORS[dominant[0]] || "#888") : "#888";

    html += `
      <div class="pop-row">
        <span class="pop-label">${sit}</span>
        <div class="pop-bar-bg"><div class="pop-bar-fill" style="width:${pct}%;background:${color}"></div></div>
        <span class="pop-pct">${pct}%</span>
      </div>
    `;
  }
  el.innerHTML = html;
}

// ---------------------------------------------------------------------------
// Divergence panel
// ---------------------------------------------------------------------------

function updateDivergence(creatures) {
  const el = document.getElementById("divergence-panel");
  const withRetrieval = creatures.filter(c => c.last_retrieval && c.last_retrieval.text);
  if (withRetrieval.length === 0) return;

  let html = "";
  for (const c of withRetrieval.slice(0, 6)) {
    const r = c.last_retrieval;
    const hue = c.appearance?.primary_hue || 0;
    const genes = (r.all_genes || [r.gene]).filter(Boolean);
    const geneTags = genes.map(g => `<span class="div-gene">${g}</span>`).join(" ");
    const locus = r.locus ? `<span class="div-locus">${r.locus}</span>` : "";
    html += `
      <div class="div-entry">
        <div class="div-header">
          <span class="color-swatch" style="background:hsl(${hue},70%,55%);width:8px;height:8px"></span>
          <strong style="font-size:11px">${c.name}</strong>
          <span class="div-trigger">${r.trigger}</span>
        </div>
        <div class="div-genes">${geneTags}</div>
        ${locus}
        <div class="div-text">${r.text}</div>
        <div class="div-score">Score: ${r.score.toFixed(3)}</div>
      </div>
    `;
  }
  el.innerHTML = html;
}

// ---------------------------------------------------------------------------
// Conversation log
// ---------------------------------------------------------------------------

function addConvEntry(text, speaker, color, type) {
  const el = document.getElementById("conv-log");
  const entry = document.createElement("div");
  entry.className = `conv-entry ${type || ""}`;

  if (speaker) {
    entry.innerHTML = `
      <div class="conv-header">
        <span class="conv-speaker-dot" style="background:${color || '#888'}"></span>
        <span>${speaker}</span>
      </div>
      <div class="conv-text">${text}</div>
    `;
  } else {
    entry.innerHTML = `<div class="conv-text">${text}</div>`;
  }

  el.prepend(entry);
  while (el.children.length > 50) el.lastChild.remove();
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

document.getElementById("btn-pause").addEventListener("click", () => send({ type: "toggle_pause" }));

document.getElementById("btn-bear").addEventListener("click", () => {
  bearEnabled = !bearEnabled;
  send({ type: "bear_set", enabled: bearEnabled });
  const btn = document.getElementById("btn-bear");
  btn.textContent = bearEnabled ? "Disable BEAR" : "Enable BEAR";
  btn.className = bearEnabled ? "bear-on" : "bear-off";
});

document.getElementById("btn-names").addEventListener("click", () => {
  namesVisible = !namesVisible;
  const btn = document.getElementById("btn-names");
  btn.textContent = namesVisible ? "Names On" : "Names Off";
  btn.className = namesVisible ? "names-on" : "names-off";
});

document.getElementById("speed-slider").addEventListener("input", (e) => {
  send({ type: "speed", value: parseFloat(e.target.value) });
});

// Genetics controls
document.getElementById("sel-recombination")?.addEventListener("change", (e) => {
  send({ type: "set_breeding", recombination: e.target.value });
});
document.getElementById("sel-ploidy")?.addEventListener("change", (e) => {
  send({ type: "set_breeding", ploidy: e.target.value });
});
document.getElementById("chk-auto-breed")?.addEventListener("change", (e) => {
  send({ type: "autonomous_breeding", enabled: e.target.checked });
});
document.getElementById("chk-auto-regulate")?.addEventListener("change", (e) => {
  const row = document.getElementById("auto-regulate-target-row");
  if (row) row.style.display = e.target.checked ? "flex" : "none";
  send({ type: "auto_regulate", enabled: e.target.checked,
         target: parseInt(document.getElementById("auto-regulate-target")?.value || "20") });
});
document.getElementById("auto-regulate-target")?.addEventListener("input", (e) => {
  const val = parseInt(e.target.value);
  document.getElementById("auto-regulate-target-val").textContent = val;
  if (document.getElementById("chk-auto-regulate")?.checked) {
    send({ type: "auto_regulate", enabled: true, target: val });
  }
});
const restartBtn = document.getElementById("btn-restart");
if (restartBtn) {
  restartBtn.addEventListener("click", () => {
    console.log("Restart clicked, ws state:", ws?.readyState);
    send({ type: "restart" });
  });
} else {
  console.error("btn-restart not found in DOM");
}

// Tuning amplifiers
document.querySelectorAll(".tune-row input[data-param]").forEach(slider => {
  slider.addEventListener("input", () => {
    const val = parseFloat(slider.value);
    send({ type: "amplifier", param: slider.dataset.param, value: val });
    const label = document.querySelector(`.tune-val[data-for="${slider.dataset.param}"]`);
    if (label) label.textContent = `${val.toFixed(1)}×`;
  });
});

// Simulation parameters
document.querySelectorAll(".tune-row input[data-simparam]").forEach(slider => {
  slider.addEventListener("input", () => {
    const val = parseFloat(slider.value);
    const param = slider.dataset.simparam;
    send({ type: "sim_param", param, value: val });
    const label = document.querySelector(`.tune-val[data-for-sim="${param}"]`);
    if (label) {
      const suffix = ["breed_cooldown", "predator_interval"].includes(param) ? "s" : "";
      label.textContent = `${val}${suffix}`;
    }
  });
});

// ---------------------------------------------------------------------------
// Weather effects
// ---------------------------------------------------------------------------

function applyWeather(weather) {
  if (weather === currentWeather) return;
  currentWeather = weather;
  const s = WEATHER_SETTINGS[weather] || WEATHER_SETTINGS.mild;

  // Sky and lighting transitions
  scene.background = new THREE.Color(s.sky);
  ambient.color.set(s.ambient);
  ambient.intensity = s.ambientI;
  sun.intensity = s.sunI;
  groundMat.color.set(s.ground);

  // Fog
  if (s.fog) {
    scene.fog = new THREE.FogExp2(s.fog, weather === "storm" ? 0.04 : 0.02);
  } else {
    scene.fog = null;
  }

  // Clear old particles
  for (const p of weatherParticles) scene.remove(p);
  weatherParticles = [];

  // Create weather particles
  if (weather === "storm") {
    // Rain
    const rainGeo = new THREE.BufferGeometry();
    const rainVerts = [];
    for (let i = 0; i < 300; i++) {
      rainVerts.push(Math.random() * 20, Math.random() * 8 + 2, Math.random() * 14);
    }
    rainGeo.setAttribute("position", new THREE.Float32BufferAttribute(rainVerts, 3));
    const rainMat = new THREE.PointsMaterial({ color: 0x8899bb, size: 0.06, transparent: true, opacity: 0.6 });
    const rain = new THREE.Points(rainGeo, rainMat);
    rain.userData.type = "rain";
    scene.add(rain);
    weatherParticles.push(rain);
  } else if (weather === "cold") {
    // Snow
    const snowGeo = new THREE.BufferGeometry();
    const snowVerts = [];
    for (let i = 0; i < 150; i++) {
      snowVerts.push(Math.random() * 20, Math.random() * 6 + 1, Math.random() * 14);
    }
    snowGeo.setAttribute("position", new THREE.Float32BufferAttribute(snowVerts, 3));
    const snowMat = new THREE.PointsMaterial({ color: 0xffffff, size: 0.08, transparent: true, opacity: 0.8 });
    const snow = new THREE.Points(snowGeo, snowMat);
    snow.userData.type = "snow";
    scene.add(snow);
    weatherParticles.push(snow);
  } else if (weather === "heat") {
    // Heat shimmer — subtle floating particles
    const heatGeo = new THREE.BufferGeometry();
    const heatVerts = [];
    for (let i = 0; i < 60; i++) {
      heatVerts.push(Math.random() * 20, Math.random() * 2 + 0.3, Math.random() * 14);
    }
    heatGeo.setAttribute("position", new THREE.Float32BufferAttribute(heatVerts, 3));
    const heatMat = new THREE.PointsMaterial({ color: 0xffcc44, size: 0.1, transparent: true, opacity: 0.3 });
    const heat = new THREE.Points(heatGeo, heatMat);
    heat.userData.type = "heat";
    scene.add(heat);
    weatherParticles.push(heat);
  }
}

function animateWeather() {
  for (const particles of weatherParticles) {
    const pos = particles.geometry.attributes.position;
    const arr = pos.array;
    const type = particles.userData.type;
    for (let i = 0; i < arr.length; i += 3) {
      if (type === "rain") {
        arr[i + 1] -= 0.15;  // fall fast
        arr[i] += (Math.random() - 0.5) * 0.02;  // slight wind
        if (arr[i + 1] < 0) { arr[i + 1] = 8 + Math.random() * 2; arr[i] = Math.random() * 20; }
      } else if (type === "snow") {
        arr[i + 1] -= 0.02;  // fall slowly
        arr[i] += Math.sin(performance.now() / 1000 + i) * 0.005;  // drift
        arr[i + 2] += Math.cos(performance.now() / 1200 + i) * 0.003;
        if (arr[i + 1] < 0) { arr[i + 1] = 6 + Math.random(); arr[i] = Math.random() * 20; }
      } else if (type === "heat") {
        arr[i + 1] += 0.008;  // rise slowly
        arr[i] += Math.sin(performance.now() / 800 + i) * 0.008;  // shimmer
        if (arr[i + 1] > 3) { arr[i + 1] = 0.3; arr[i] = Math.random() * 20; }
      }
    }
    pos.needsUpdate = true;
  }
}

// ---------------------------------------------------------------------------
// Animation loop
// ---------------------------------------------------------------------------

function animate() {
  requestAnimationFrame(animate);

  // Weather particles
  animateWeather();

  // Rotate creatures slightly
  for (const [id, mesh] of Object.entries(creatureMeshes)) {
    const bc = mesh.userData.bodyCore;
    if (bc) {
      // Gentle breathing animation
      const t = performance.now() / 1000;
      const breathe = 1 + Math.sin(t * 1.5 + mesh.position.x) * 0.02;
      bc.scale.set(breathe, breathe, breathe);
    }
  }

  renderer.render(scene, camera);
}

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

connect();
animate();
