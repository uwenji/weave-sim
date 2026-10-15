// three.js view of a WeaveSim model (Z up): the carrier mesh, the laths as swept rectangular sections on the current
// solver state (node frames: X = width, Y = thickness / surface normal, Z = tangent), centrelines, crossings (knots),
// point loads, anchors, the selection and the UTM probes; result layers as colours along the laths (a scalar per lath
// vertex through a colour-map texture, so the gradient is exact and smooth); member diagrams, joints (discretisation),
// the highlighted laths of a lath selection (LocalSH idiom); a soft ground shadow under the model (Polyscope's
// "shadow_only" ground plane). Navigation cube, canonical views, turntable and projection switch as in LocalSH and the
// Joint FEA app. Draws only when something changed.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { NavCube } from './navcube.js';
import { nodeQuats } from './model.js';
import { toFrame } from './quat.js';
import {
  lutColor, lutBytes, toLinear, familyColor, STRAND_COLORS, GROUP_COLORS, TIMBER, OVER_COLOR, UNDER_COLOR, PIECE_COLORS, CUT_COLOR,
  KNOT, KNOT_BOUNDARY, KNOT_FLIP, KNOT_CONFLICT, LOAD, ANCHOR_FULL, ANCHOR_PIN, ANCHOR_PART, DIAG_OUTLINE, hex,
} from './colormap.js';
import { fkey } from './topology.js';
import { interp } from './fields.js';

const cssVar = (name, fallback) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
const cssColor = (name, fallback) => new THREE.Color(cssVar(name, fallback));
const HOME_DIR = new THREE.Vector3(-0.85, -1.15, 0.7).normalize();
const POLE_EPS = 2e-3;
const SEL_HEX = '#EE7733', SEL_RGB = hex(SEL_HEX);
const SUN_DIR = new THREE.Vector3(0.22, 0.32, 1).normalize();     // shadow light: from above, slightly offset
const lin = (rgb) => [toLinear(rgb[0]), toLinear(rgb[1]), toLinear(rgb[2])];
const srgbColor = (rgb) => new THREE.Color().setRGB(rgb[0], rgb[1], rgb[2], THREE.SRGBColorSpace);
/** a CSS colour or a THREE.Color → [r, g, b] sRGB */
const toSRGB = (c) => { const t = {}; (c.isColor ? c : new THREE.Color(c)).getRGB(t, THREE.SRGBColorSpace); return [t.r, t.g, t.b]; };

/** MeshStandardMaterial whose diffuse colour is a colour map looked up per fragment from a per-vertex scalar `scal`
 *  (0…1): the gradient along a lath is the colour map itself, not an RGB blend of its ends; `bands` > 0 quantises */
function lutMaterial(tex) {
  const mat = new THREE.MeshStandardMaterial({ roughness: 0.78, metalness: 0, side: THREE.DoubleSide });
  mat.userData.uniforms = { lutTex: { value: tex }, bands: { value: 0 } };
  mat.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, mat.userData.uniforms);
    sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nattribute float scal;\nvarying float vScal;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvScal = scal;');
    sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\nuniform sampler2D lutTex;\nuniform float bands;\nvarying float vScal;')
      .replace('#include <color_fragment>', '#include <color_fragment>\n{ float t = clamp(vScal, 0.0, 1.0); if (bands > 0.5) t = (floor(min(t, 0.99999) * bands) + 0.5) / bands; diffuseColor.rgb = texture2D(lutTex, vec2((t * 255.0 + 0.5) / 256.0, 0.5)).rgb; }');
  };
  mat.customProgramCacheKey = () => 'weavesim-lut';
  return mat;
}

export class View {
  constructor(canvas, labelsEl = null) {
    this.canvas = canvas; this.labelsEl = labelsEl;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, powerPreference: 'high-performance', preserveDrawingBuffer: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.shadowMap.enabled = true; this.renderer.shadowMap.type = THREE.VSMShadowMap;
    this.scene = new THREE.Scene();
    this.persp = new THREE.PerspectiveCamera(38, 1, 0.01, 1000); this.persp.up.set(0, 0, 1);
    this.orthoH = 1;
    this.ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 1000); this.ortho.up.set(0, 0, 1);
    this.camera = this.persp; this.projection = 'persp';
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true; this.controls.dampingFactor = 0.12; this.controls.rotateSpeed = 0.7;
    // a wheel zoom moves the camera inside the event handler (OrbitControls.update runs there), not in the render
    // loop: listen for every change, or the zoomed frame waits for the next orbit
    this.controls.addEventListener('change', () => { this.needsRender = true; });
    this.clock = new THREE.Clock(); this.anim = null; this.turntable = false; this.turnSpeed = 12;
    // the home view is fitted to the canvas aspect; until the user moves the camera it is refitted when the canvas changes
    // size (the page settling its layout after load, the Display sidebar folding, the window resized)
    this.autoFit = false;
    canvas.addEventListener('pointerdown', () => { this.anim = null; this.autoFit = false; });
    canvas.addEventListener('wheel', () => { this.autoFit = false; }, { passive: true });
    this.hemi = new THREE.HemisphereLight(0xffffff, 0x9a9ea8, 1.35); this.scene.add(this.hemi);
    this.key = new THREE.DirectionalLight(0xffffff, 1.25); this.key.position.set(-1.2, -1.6, 2.2); this.scene.add(this.key);
    this.fill = new THREE.DirectionalLight(0xffffff, 0.6); this.fill.position.set(1.5, 0.8, 0.6); this.scene.add(this.fill);
    this.sun = new THREE.DirectionalLight(0xffffff, 0.55); this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048); this.sun.shadow.radius = 7; this.sun.shadow.blurSamples = 16; this.sun.shadow.bias = -0.0005;
    this.scene.add(this.sun, this.sun.target);
    this.ground = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.ShadowMaterial({ color: 0x000000, opacity: 0.2, transparent: true, depthWrite: false }));
    this.ground.receiveShadow = true; this.ground.renderOrder = -2; this.scene.add(this.ground);
    this.group = new THREE.Group(); this.scene.add(this.group);
    this.layers = {};
    this.visible = { mesh: true, wire: false, laths: true, lines: false, knots: true, loads: true, anchors: true, selection: true, probes: true, frames: false, shadow: true, diagram: true, joints: true, path: true };
    this.raycaster = new THREE.Raycaster();
    this.deform = 1.0; this.needsRender = true;
    this.colouring = { mode: 'family' };
    this.bboxSize = 1; this.center = new THREE.Vector3(); this.size = 1;
    this.labelCache = new Map();
    this.labelItems = [];
    this.strandSel = new Set();
    this.lutTex = new THREE.DataTexture(lutBytes('viridis'), 256, 1, THREE.RGBAFormat);
    this.lutTex.colorSpace = THREE.SRGBColorSpace; this.lutTex.magFilter = THREE.LinearFilter; this.lutTex.minFilter = THREE.LinearFilter; this.lutTex.needsUpdate = true;
    this.lutName = 'viridis';
    this.matVC = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.78, metalness: 0, side: THREE.DoubleSide });
    this.matLut = lutMaterial(this.lutTex);
    this.applyTheme();
    this._ro = new ResizeObserver(() => this.resize()); this._ro.observe(canvas.parentElement);
    this.resize();
  }

  applyTheme() {
    this.dark = cssVar('--scheme', 'light') === 'dark';
    this.renderer.setClearColor(cssColor('--canvas', '#f6f6f5'), 1);
    this.colCanvas = cssColor('--canvas', '#f6f6f5');
    this.colMesh = cssColor('--mesh', '#e0e0de'); this.colEdge = cssColor('--mesh-edge', '#b8b8b5');
    this.colAccent = cssColor('--accent', '#2973c7'); this.colFg = cssColor('--fg', '#212324'); this.colWarn = cssColor('--warn', '#c26a1c');
    this.ground.material.opacity = this.dark ? 0.45 : 0.26;
    if (this.layers.mesh) this.layers.mesh.material.color.copy(this.colMesh);
    if (this.layers.wire) this.layers.wire.material.color.copy(this.colEdge);
    if (this.layers.path) this.layers.path.traverse(o => { if (o.material) o.material.color.copy(this.colFg); });
    this.labelCache.clear();
    if (this.nav) this.nav.applyTheme();
    if (this.st) { this.recolour(); this.setOverlays(this.overlays || {}); if (this.diagramSpec) this.setDiagram(this.diagramSpec); if (this.jointSpec) this.setJoints(this.jointSpec); }
    this.needsRender = true;
  }

  resize() {
    const el = this.canvas.parentElement, w = el.clientWidth || 800, h = el.clientHeight || 600, aspect = w / h;
    const refit = this.autoFit && this.st && Math.abs(aspect - this.persp.aspect) > 0.02 * this.persp.aspect;
    this.renderer.setSize(w, h, false);
    this.persp.aspect = aspect; this.persp.updateProjectionMatrix();
    this.ortho.left = -this.orthoH * aspect; this.ortho.right = this.orthoH * aspect; this.ortho.top = this.orthoH; this.ortho.bottom = -this.orthoH; this.ortho.updateProjectionMatrix();
    this.scene.traverse(o => { if (o.material && o.material.isLineMaterial) o.material.resolution.set(w, h); });
    if (this.nav) this.nav.resize();
    if (refit) this.home(0);
    this.needsRender = true;
  }

  /** draws only when something changed (camera, geometry, colours, overlays) — a laptop GPU idles otherwise */
  render() {
    const dt = Math.min(this.clock.getDelta(), 0.5);
    const animating = !!this.anim;
    this._stepAnim();
    const moved = this.controls.update(dt);
    if (moved || animating || this.needsRender) { this.renderer.render(this.scene, this.camera); this.needsRender = false; this._placeLabels(); }
    if (this.nav) this.nav.render(this.camera, this.controls.target);
  }
  markDirty() { this.needsRender = true; }

  // --- navigation -------------------------------------------------------------------------------------------------
  attachNav(canvas) {
    this.nav = new NavCube(canvas, {
      onPick: (dir) => { this.autoFit = false; this.lookFrom(dir); },
      onDrag: (dx, dy) => { this.autoFit = false; const h = this.canvas.clientHeight || 600, k = 2 * Math.PI * this.controls.rotateSpeed / h; this.controls._rotateLeft(k * dx); this.controls._rotateUp(k * dy); this.needsRender = true; },
    });
  }
  viewDir() { const d = this.camera.position.clone().sub(this.controls.target); return d.lengthSq() > 1e-18 ? d.normalize() : HOME_DIR.clone(); }
  viewDist() { return this.camera.position.distanceTo(this.controls.target); }
  fitDistance() { const d = this.bboxSize, aspect = this.persp.aspect || 1; return Math.max(1.59 * d, 1.5 * d / aspect); }
  fitZoom() { const hh = this.fitDistance() * Math.tan(THREE.MathUtils.degToRad(this.persp.fov / 2)); return this.orthoH / hh; }
  lookFrom(dir, opts = {}) {
    const d = (dir.isVector3 ? dir.clone() : new THREE.Vector3().fromArray(dir)).normalize();
    if (1 - Math.abs(d.z) < 1e-6) {
      const t = opts.tiltFrom && Math.hypot(opts.tiltFrom.x, opts.tiltFrom.y) > 1e-9 ? new THREE.Vector3(opts.tiltFrom.x, opts.tiltFrom.y, 0).normalize() : new THREE.Vector3(0, -1, 0);
      d.set(t.x * POLE_EPS, t.y * POLE_EPS, Math.sign(d.z)).normalize();
    }
    this.anim = { t0: performance.now(), ms: opts.ms ?? 450, dir0: this.viewDir(), dir1: d, dist0: this.viewDist(), dist1: opts.dist ?? this.viewDist(),
      tg0: this.controls.target.clone(), tg1: opts.target ? opts.target.clone() : this.controls.target.clone(), zoom0: this.ortho.zoom, zoom1: opts.zoom ?? this.ortho.zoom };
    if (opts.ms === 0) this._stepAnim(performance.now() + 1);
  }
  _stepAnim(now = performance.now()) {
    const a = this.anim; if (!a) return;
    const t = Math.min(1, (now - a.t0) / Math.max(a.ms, 1)), e = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    const dir = slerp(a.dir0, a.dir1, e), dist = a.dist0 + (a.dist1 - a.dist0) * e;
    this.controls.target.copy(a.tg0).lerp(a.tg1, e);
    this.camera.position.copy(this.controls.target).addScaledVector(dir, dist);
    if (this.projection === 'ortho') { this.ortho.zoom = a.zoom0 + (a.zoom1 - a.zoom0) * e; this.ortho.updateProjectionMatrix(); }
    this.needsRender = true;
    if (t >= 1) this.anim = null;
  }
  orbitBy(dAz, dPol) {
    this.autoFit = false;
    const d0 = this.viewDir(), az = Math.atan2(d0.y, d0.x) + dAz;
    const pol = Math.min(Math.PI - POLE_EPS, Math.max(POLE_EPS, Math.acos(Math.max(-1, Math.min(1, d0.z))) + dPol));
    this.lookFrom(new THREE.Vector3(Math.sin(pol) * Math.cos(az), Math.sin(pol) * Math.sin(az), Math.cos(pol)), { tiltFrom: d0 });
  }
  home(ms = 500) { this.lookFrom(HOME_DIR, { dist: this.fitDistance(), target: this.center, zoom: this.fitZoom(), ms }); }
  fit(ms = 400) { this.lookFrom(this.viewDir(), { dist: this.fitDistance(), target: this.center, zoom: this.fitZoom(), ms, tiltFrom: this.viewDir() }); }
  setTurntable(on, degPerSec = this.turnSpeed) { if (on) this.autoFit = false; this.turntable = !!on; this.turnSpeed = degPerSec; this.controls.autoRotate = this.turntable; this.controls.autoRotateSpeed = degPerSec / 6; }
  setProjection(kind) {
    if (kind === this.projection) return;
    this.autoFit = false;
    const dir = this.viewDir(), tg = this.controls.target.clone(), tanHalf = Math.tan(THREE.MathUtils.degToRad(this.persp.fov / 2));
    this.anim = null;
    if (kind === 'ortho') { const hh = this.viewDist() * tanHalf; this.ortho.zoom = this.orthoH / hh; this.ortho.position.copy(this.persp.position); this.ortho.updateProjectionMatrix(); this.camera = this.ortho; }
    else { const hh = this.orthoH / this.ortho.zoom; this.persp.position.copy(tg).addScaledVector(dir, hh / tanHalf); this.camera = this.persp; }
    this.projection = kind; this.controls.object = this.camera; this.camera.lookAt(tg); this.controls.update(); this.needsRender = true;
  }

  _setLayer(name, obj) {
    this.needsRender = true;
    if (this.layers[name]) { this.group.remove(this.layers[name]); disposeObject(this.layers[name]); }
    this.layers[name] = obj;
    if (obj) { obj.visible = !!this.visible[name]; this.group.add(obj); }
  }
  setVisible(name, on) {
    this.needsRender = true; this.visible[name] = on;
    if (name === 'shadow') { this.ground.visible = on; this.sun.castShadow = on; return; }
    if (this.layers[name]) this.layers[name].visible = on;
    if (name === 'diagram') this._setLabels(on && this.diagramSpec ? this.diagramSpec.labels : []);
    if (name === 'laths' || name === 'lines') this.recolour();
  }

  // --- model --------------------------------------------------------------------------------------------------------
  /** a new model: static maps (segment → strand, section per strand, crossing node pairs), the carrier, the camera */
  setModel({ topo, st, scene }, { keepCamera = false } = {}) {
    this.topo = topo; this.st = st; this.sceneModel = scene;
    const S = st.S, m = st.m;
    this.secW = new Float32Array(S); this.secH = new Float32Array(S);
    for (let s = 0; s < S; s++) { const sec = scene.group(scene.strandGroup(s)).section; this.secW[s] = sec.width; this.secH[s] = sec.height; }
    this.strandGroup = Int32Array.from({ length: S }, (_, s) => scene.strandGroup(s));
    this.family = Int32Array.from(topo.strands.map(s => s.family ?? 0));
    this.closed = topo.strands.map(s => !!s.closed);
    this.strandSel = new Set([...this.strandSel].filter(s => s < S));
    // crossings: node pair per knot (sorted edge order = st.knotEdges), and the knot state colours (Python theme)
    const K = st.K; this.knotNodes = new Int32Array(2 * K); this.knotState = new Uint8Array(K);
    for (let c = 0; c < K; c++) {
      const kn = topo.knots.get(st.knotEdges[c]);
      this.knotNodes[2 * c] = st.nodeFrameIds.get(fkey(...kn.a)); this.knotNodes[2 * c + 1] = st.nodeFrameIds.get(fkey(...kn.b));
      this.knotState[c] = kn.conflict ? 3 : kn.flipped ? 2 : kn.isBoundary ? 1 : 0;
    }
    // over / under state per node (viewer.apply_lath_colouring): interpolated between the crossings of a strand
    this.overState = new Float32Array(st.n).fill(0.5);
    for (let s = 0; s < S; s++) {
      const rows = [], vals = [];
      topo.strands[s].edges.forEach((_, k) => { const r = topo.knotNode.get(fkey(s, k)); if (r !== undefined && topo.hasFrame(s, k)) { rows.push(r); vals.push(topo.isOver(s, k) ? 1 : 0); } });
      if (!rows.length) continue;
      const order = rows.map((r, i) => i).sort((a, b) => rows[a] - rows[b]), xr = order.map(i => rows[i]), fv = order.map(i => vals[i]);
      for (let i = st.nodeOffset[s]; i < st.nodeOffset[s + 1]; i++) this.overState[i] = interp(i - st.nodeOffset[s], xr, fv);
    }
    // bounding box of the model (the carrier when there is one: the Python viewer freezes the extents on the mesh)
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < st.n; i++) for (let c = 0; c < 3; c++) { const v = st.X[3 * i + c]; lo[c] = Math.min(lo[c], v); hi[c] = Math.max(hi[c], v); }
    this.bboxSize = Math.max(Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]), 1e-9);
    this.center = new THREE.Vector3((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2);
    const mesh = topo.mesh;
    let mlo = lo.slice(), mhi = hi.slice();
    if (mesh) { mlo = [Infinity, Infinity, Infinity]; mhi = [-Infinity, -Infinity, -Infinity]; for (let i = 0; i < mesh.V.length; i += 3) for (let c = 0; c < 3; c++) { mlo[c] = Math.min(mlo[c], mesh.V[i + c]); mhi[c] = Math.max(mhi[c], mesh.V[i + c]); } }
    this.size = Math.hypot(mhi[0] - mlo[0], mhi[1] - mlo[1], mhi[2] - mlo[2]) || 1.0;     // diagrams.py _model_size
    let wmax = 0, hmax = 0; for (let s = 0; s < S; s++) { wmax = Math.max(wmax, this.secW[s]); hmax = Math.max(hmax, this.secH[s]); }
    this.knotR = 0.32 * Math.max(wmax, 1e-9);
    // ground shadow just under the lowest point (Python: lowest mesh point − section height)
    const zg = Math.min(mlo[2], lo[2]) - (scene.section.height || hmax);
    const gs = 6 * this.bboxSize;
    this.ground.scale.set(gs, gs, 1); this.ground.position.set(this.center.x, this.center.y, zg);
    const sc = this.sun.shadow.camera, R0 = 0.78 * this.bboxSize;
    sc.left = -R0; sc.right = R0; sc.top = R0; sc.bottom = -R0; sc.near = 0.05 * this.bboxSize; sc.far = 6 * this.bboxSize; sc.updateProjectionMatrix();
    this.sun.position.copy(this.center).addScaledVector(SUN_DIR, 2.5 * this.bboxSize); this.sun.target.position.copy(this.center); this.sun.target.updateMatrixWorld();
    this.sun.shadow.needsUpdate = true;
    // carrier mesh
    if (mesh) {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(Float32Array.from(mesh.V), 3));
      g.setIndex(new THREE.BufferAttribute(Uint32Array.from(mesh.F), 1));
      g.computeVertexNormals();
      const surf = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color: this.colMesh, roughness: 0.95, metalness: 0, side: THREE.DoubleSide, transparent: true, opacity: 0.3, depthWrite: false, polygonOffset: true, polygonOffsetFactor: 2, polygonOffsetUnits: 2 }));
      surf.userData.kind = 'mesh'; surf.renderOrder = -1;
      this._setLayer('mesh', surf);
      this._setLayer('wire', new THREE.LineSegments(new THREE.WireframeGeometry(g), new THREE.LineBasicMaterial({ color: this.colEdge, transparent: true, opacity: 0.5 })));
    } else { this._setLayer('mesh', null); this._setLayer('wire', null); }
    // laths: 4 faces × 4 vertices per segment (per-face normals); colour and colour-map scalar per vertex
    const lg = new THREE.BufferGeometry();
    this.lathPos = new Float32Array(m * 16 * 3); this.lathNrm = new Float32Array(m * 16 * 3); this.lathCol = new Float32Array(m * 16 * 3); this.lathScal = new Float32Array(m * 16);
    const idx = new Uint32Array(m * 24);
    // winding (v, v+2, v+1): the geometric normal of each face points along its explicit normal (frames are right-handed)
    for (let j = 0; j < m; j++) for (let f = 0; f < 4; f++) { const v = 16 * j + 4 * f, o = 24 * j + 6 * f; idx[o] = v; idx[o + 1] = v + 2; idx[o + 2] = v + 1; idx[o + 3] = v; idx[o + 4] = v + 3; idx[o + 5] = v + 2; }
    lg.setAttribute('position', new THREE.BufferAttribute(this.lathPos, 3).setUsage(THREE.DynamicDrawUsage));
    lg.setAttribute('normal', new THREE.BufferAttribute(this.lathNrm, 3).setUsage(THREE.DynamicDrawUsage));
    lg.setAttribute('color', new THREE.BufferAttribute(this.lathCol, 3).setUsage(THREE.DynamicDrawUsage));
    lg.setAttribute('scal', new THREE.BufferAttribute(this.lathScal, 1).setUsage(THREE.DynamicDrawUsage));
    lg.setIndex(new THREE.BufferAttribute(idx, 1));
    const laths = new THREE.Mesh(lg, this.matVC);
    laths.userData.kind = 'lath'; laths.castShadow = true; laths.userData.sharedMaterial = true;
    this._setLayer('laths', laths);
    // centrelines
    const cg = new THREE.BufferGeometry();
    this.linePos = new Float32Array(m * 6); this.lineCol = new Float32Array(m * 6);
    cg.setAttribute('position', new THREE.BufferAttribute(this.linePos, 3).setUsage(THREE.DynamicDrawUsage));
    cg.setAttribute('color', new THREE.BufferAttribute(this.lineCol, 3).setUsage(THREE.DynamicDrawUsage));
    this._setLayer('lines', new THREE.LineSegments(cg, new THREE.LineBasicMaterial({ vertexColors: true })));
    // knots
    if (K) {
      const kg = new THREE.SphereGeometry(1, 14, 10);
      const km = new THREE.InstancedMesh(kg, new THREE.MeshStandardMaterial({ roughness: 0.6, metalness: 0 }), K);
      km.instanceMatrix.setUsage(THREE.DynamicDrawUsage); km.userData.kind = 'knot'; km.castShadow = true;
      for (let c = 0; c < K; c++) km.setColorAt(c, new THREE.Color());
      this._setLayer('knots', km);
    } else this._setLayer('knots', null);
    this.diagramSpec = null; this._setLayer('diagram', null); this._setLabels([]);
    this.jointSpec = null; this._setLayer('joints', null);
    this.updateGeometry(st);
    this.recolour();
    for (const cam of [this.persp, this.ortho]) { cam.near = this.bboxSize * 0.002; cam.far = this.bboxSize * 40; cam.updateProjectionMatrix(); }
    if (!keepCamera) { this.home(0); this.autoFit = true; }
  }

  /** positions on the current state (st.X, st.Q); displayed geometry = X0 + deform · (X − X0) */
  updateGeometry(st = this.st) {
    if (!st || !this.lathPos) return;
    this.st = st;
    const n = st.n, m = st.m, X = st.X, X0 = st.X0, s = this.deform;
    const P = this.dispX && this.dispX.length === 3 * n ? this.dispX : (this.dispX = new Float64Array(3 * n));
    if (s === 1) P.set(X); else for (let i = 0; i < 3 * n; i++) P[i] = X0[i] + s * (X[i] - X0[i]);
    const nq = nodeQuats(st, this.nq && this.nq.length === 4 * n ? this.nq : (this.nq = new Float64Array(4 * n)));
    const R = this.nodeR && this.nodeR.length === 9 * n ? this.nodeR : (this.nodeR = new Float64Array(9 * n)), tmp = new Float64Array(9);
    for (let i = 0; i < n; i++) { toFrame(nq, 4 * i, tmp); R.set(tmp, 9 * i); }
    const pos = this.lathPos, nrm = this.lathNrm, lp = this.linePos;
    const corner = (o, i, sx, sy, w, h) => { for (let c = 0; c < 3; c++) pos[o + c] = P[3 * i + c] + sx * 0.5 * w * R[9 * i + c] + sy * 0.5 * h * R[9 * i + 3 + c]; };
    for (let j = 0; j < m; j++) {
      const a = st.segI0[j], b = st.segI1[j], str = st.segStrand[j], w = this.secW[str], h = this.secH[str];
      const base = 48 * j;
      // faces: +Y (top), −Y, +X, −X; vertices a−, a+, b+, b− around each face
      corner(base + 0, a, -1, 1, w, h); corner(base + 3, a, 1, 1, w, h); corner(base + 6, b, 1, 1, w, h); corner(base + 9, b, -1, 1, w, h);
      corner(base + 12, a, 1, -1, w, h); corner(base + 15, a, -1, -1, w, h); corner(base + 18, b, -1, -1, w, h); corner(base + 21, b, 1, -1, w, h);
      corner(base + 24, a, 1, 1, w, h); corner(base + 27, a, 1, -1, w, h); corner(base + 30, b, 1, -1, w, h); corner(base + 33, b, 1, 1, w, h);
      corner(base + 36, a, -1, -1, w, h); corner(base + 39, a, -1, 1, w, h); corner(base + 42, b, -1, 1, w, h); corner(base + 45, b, -1, -1, w, h);
      for (let c = 0; c < 3; c++) {
        const ny = R[9 * a + 3 + c] + R[9 * b + 3 + c], nx = R[9 * a + c] + R[9 * b + c];
        for (let k = 0; k < 4; k++) { nrm[base + 3 * k + c] = ny; nrm[base + 12 + 3 * k + c] = -ny; nrm[base + 24 + 3 * k + c] = nx; nrm[base + 36 + 3 * k + c] = -nx; }
      }
      for (let c = 0; c < 3; c++) { lp[6 * j + c] = P[3 * a + c]; lp[6 * j + 3 + c] = P[3 * b + c]; }
    }
    const L = this.layers.laths.geometry;
    L.attributes.position.needsUpdate = true; L.attributes.normal.needsUpdate = true; L.computeBoundingSphere();
    const C = this.layers.lines.geometry; C.attributes.position.needsUpdate = true; C.computeBoundingSphere();
    const km = this.layers.knots;
    if (km) {
      const M4 = new THREE.Matrix4(), r = this.knotR;
      for (let c = 0; c < st.K; c++) {
        const ia = this.knotNodes[2 * c], ib = this.knotNodes[2 * c + 1];
        M4.makeScale(r, r, r).setPosition(0.5 * (P[3 * ia] + P[3 * ib]), 0.5 * (P[3 * ia + 1] + P[3 * ib + 1]), 0.5 * (P[3 * ia + 2] + P[3 * ib + 2]));
        km.setMatrixAt(c, M4);
      }
      km.instanceMatrix.needsUpdate = true; km.computeBoundingSphere();
    }
    if (this.overlays) this._placeOverlays();
    if (this.strandSel.size) this._buildPath();
    this.sun.shadow.needsUpdate = true;
    this.needsRender = true;
  }

  setDeform(s) { this.deform = s; this.updateGeometry(); }

  // --- colouring ----------------------------------------------------------------------------------------------------
  /** colouring: {mode: 'family'|'strand'|'group'|'plain'|'overunder'|'pieces', color: [r,g,b] (plain), pieces:
   *  Int32Array per node} or {mode: 'layer', where: 'node'|'segment'|'knot', values, cmap, lo, hi, bands, categorical:
   *  [hex…]} — colours sRGB */
  setColouring(c) { this.colouring = c; this.recolour(); }
  setStrandSelection(strands) { this.strandSel = new Set(strands); this._buildPath(); this.recolour(); }

  recolour() {
    const st = this.st; if (!st || !this.lathCol) return;
    const c = this.colouring || { mode: 'family' }, m = st.m, col = this.lathCol, sc = this.lathScal, lc = this.lineCol;
    const dimOthers = this.strandSel.size > 0, canvas = toSRGB(this.colCanvas);
    const layer = c.mode === 'layer', lutPath = layer && c.where !== 'knot' && !c.categorical && !dimOthers;
    const span = Math.max(c.hi - c.lo, 1e-300);
    const t01 = (v) => (Number.isFinite(v) ? (v - c.lo) / span : 0);
    const neutral = this.dark ? [0.42, 0.43, 0.45] : [0.80, 0.80, 0.78];
    const lift = (rgb) => (this.dark && !layer ? rgb.map(v => 0.82 * v + 0.18) : rgb);
    const vcol = (v) => {
      if (c.categorical) return hex(c.categorical[Math.round(v)] || '#888888');
      let t = t01(v); if (c.bands > 0) t = (Math.floor(Math.min(Math.max(t, 0), 0.99999) * c.bands) + 0.5) / c.bands;
      return lutColor(c.cmap || 'viridis', t);
    };
    const strandRGB = (s) => {
      if (c.mode === 'strand') return STRAND_COLORS[s % STRAND_COLORS.length];
      if (c.mode === 'group') return GROUP_COLORS[(this.strandGroup[s] - 1) % GROUP_COLORS.length];
      if (c.mode === 'family') return familyColor(this.family[s]);
      if (c.mode === 'plain') return c.color || TIMBER;
      return TIMBER;
    };
    // knot layers: the crossing nodes carry the colour, fading to neutral along the adjacent segments
    let knotNodeCol = null;
    if (layer && c.where === 'knot' && c.values) {
      knotNodeCol = new Map();
      for (let k = 0; k < st.K; k++) { const rgb = vcol(c.values[k]); knotNodeCol.set(this.knotNodes[2 * k], rgb); knotNodeCol.set(this.knotNodes[2 * k + 1], rgb); }
    }
    const nodeRGB = (i, s) => {
      if (layer) {
        if (knotNodeCol) return knotNodeCol.get(i) || neutral;
        if (c.where === 'node') return vcol(c.values[i]);
        return null;
      }
      if (c.mode === 'overunder') return lift(this.overState[i] > 0.5 ? OVER_COLOR : UNDER_COLOR);
      if (c.mode === 'pieces' && c.pieces) return lift(PIECE_COLORS[c.pieces[i] % 2]);
      return lift(strandRGB(s));
    };
    for (let j = 0; j < m; j++) {
      const a = st.segI0[j], b = st.segI1[j], s = st.segStrand[j];
      let ra, rb, ta = 0, tb = 0;
      if (layer && c.where === 'segment') { ra = rb = vcol(c.values[j]); ta = tb = t01(c.values[j]); }
      else if (layer && c.where === 'node') { ra = nodeRGB(a, s); rb = nodeRGB(b, s); ta = t01(c.values[a]); tb = t01(c.values[b]); }
      else { ra = nodeRGB(a, s); rb = nodeRGB(b, s); }
      if (dimOthers) {                                // a lath selection: the selected laths in the selection colour, the others faded
        if (this.strandSel.has(s)) ra = rb = SEL_RGB;
        else { ra = ra.map((v, k) => 0.42 * v + 0.58 * canvas[k]); rb = rb.map((v, k) => 0.42 * v + 0.58 * canvas[k]); }
      }
      const la = lin(ra), lb = lin(rb);
      for (let f = 0; f < 4; f++) for (let k = 0; k < 4; k++) {
        const end = k < 2, rgb = end ? la : lb, o = 48 * j + 12 * f + 3 * k;
        col[o] = rgb[0]; col[o + 1] = rgb[1]; col[o + 2] = rgb[2]; sc[16 * j + 4 * f + k] = end ? ta : tb;
      }
      for (let q = 0; q < 3; q++) { lc[6 * j + q] = la[q]; lc[6 * j + 3 + q] = lb[q]; }
    }
    const laths = this.layers.laths;
    if (lutPath) {
      if (this.lutName !== c.cmap) { this.lutTex.image.data.set(lutBytes(c.cmap || 'viridis')); this.lutTex.needsUpdate = true; this.lutName = c.cmap; }
      this.matLut.userData.uniforms.bands.value = c.bands || 0;
      laths.material = this.matLut;
    } else laths.material = this.matVC;
    laths.geometry.attributes.color.needsUpdate = true; laths.geometry.attributes.scal.needsUpdate = true;
    this.layers.lines.geometry.attributes.color.needsUpdate = true;
    const km = this.layers.knots;
    if (km) {
      const states = [KNOT, KNOT_BOUNDARY, KNOT_FLIP, KNOT_CONFLICT].map(v => (this.dark && v === KNOT ? [0.80, 0.80, 0.82] : v));
      for (let k = 0; k < st.K; k++) km.setColorAt(k, srgbColor(layer && c.where === 'knot' && c.values ? vcol(c.values[k]) : states[this.knotState[k]]));
      if (km.instanceColor) km.instanceColor.needsUpdate = true;
    }
    this.needsRender = true;
  }

  // --- overlays: loads, anchors, selection, probes ------------------------------------------------------------------
  /** overlays: {loads: [{node, F: [x,y,z] world}], anchors: [{node, fix: [6]}], selection: [node], probes: [{node, axis,
   *  label, color}], frames: bool, arrowScale} */
  setOverlays(o) { this.overlays = o; this._placeOverlays(); }

  _placeOverlays() {
    const o = this.overlays || {}, P = this.dispX, d = this.bboxSize;
    if (!P) return;
    const pos = (i) => new THREE.Vector3(P[3 * i], P[3 * i + 1], P[3 * i + 2]);
    // loads: arrows pointing at the node
    const loads = (o.loads || []).filter(l => Math.hypot(...l.F) > 0);
    if (loads.length) {
      const fmax = Math.max(...loads.map(l => Math.hypot(...l.F))), scale = o.arrowScale ?? 1, from = [], vec = [];
      for (const l of loads) {
        const f = Math.hypot(...l.F), len = scale * d * 0.07 * (0.35 + 0.65 * f / fmax), dir = l.F.map(v => v / f), tip = pos(l.node);
        from.push(tip.x - len * dir[0], tip.y - len * dir[1], tip.z - len * dir[2]); vec.push(len * dir[0], len * dir[1], len * dir[2]);
      }
      this._setLayer('loads', this._arrows(from, vec, LOAD, 0.0028 * d * Math.sqrt(scale)));
    } else this._setLayer('loads', null);
    // anchors: cubes (fully fixed) / pyramids (pinned or partial), coloured by fixity as in the Python viewer
    const an = o.anchors || [];
    if (an.length) {
      const s = Math.max(this.knotR * 1.6, d * 0.004);
      const g = new THREE.Group();
      const geoBox = new THREE.BoxGeometry(s, s, s), geoTet = new THREE.ConeGeometry(0.75 * s, 1.3 * s, 4);
      const kind = (a) => { const t = a.fix.slice(0, 3).filter(Boolean).length, r = a.fix.slice(3).filter(Boolean).length; return t === 3 && r === 3 ? 0 : t === 3 ? 1 : 2; };
      const mk = (list, geo, rgb, rotX) => {
        if (!list.length) return;
        const im = new THREE.InstancedMesh(geo, new THREE.MeshStandardMaterial({ color: srgbColor(rgb), roughness: 0.6 }), list.length), M4 = new THREE.Matrix4(), q = new THREE.Quaternion();
        if (rotX) q.setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2);
        list.forEach((a, k) => { const p = pos(a.node); if (rotX) p.z -= 0.65 * s; M4.compose(p, q, new THREE.Vector3(1, 1, 1)); im.setMatrixAt(k, M4); });
        im.castShadow = true; g.add(im);
      };
      mk(an.filter(a => kind(a) === 0), geoBox, ANCHOR_FULL, false);
      mk(an.filter(a => kind(a) === 1), geoTet, ANCHOR_PIN, true);
      mk(an.filter(a => kind(a) === 2), geoTet.clone(), ANCHOR_PART, true);
      this._setLayer('anchors', g);
    } else this._setLayer('anchors', null);
    // selection
    const sel = o.selection || [];
    if (sel.length) {
      const im = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 16, 12), new THREE.MeshBasicMaterial({ color: new THREE.Color(SEL_HEX) }), sel.length), M4 = new THREE.Matrix4(), r = 1.45 * this.knotR;
      sel.forEach((i, k) => { const p = pos(i); M4.makeScale(r, r, r).setPosition(p.x, p.y, p.z); im.setMatrixAt(k, M4); });
      im.renderOrder = 3;
      this._setLayer('selection', im);
    } else this._setLayer('selection', null);
    // probes
    const pr = o.probes || [];
    if (pr.length) {
      const g = new THREE.Group();
      for (const p of pr) {
        const dir = new THREE.Vector3(...p.axis).normalize(), len = d * 0.09, tip = pos(p.node), start = tip.clone().addScaledVector(dir, -len);
        const color = p.color || cssVar('--accent', '#2973c7');
        g.add(this._arrows([start.x, start.y, start.z], [dir.x * len, dir.y * len, dir.z * len], toSRGB(color), 0.003 * d));
        if (p.label) g.add(this._label(p.label, color, start.clone().addScaledVector(dir, -0.18 * len), 0.045 * d));
      }
      this._setLayer('probes', g);
    } else this._setLayer('probes', null);
    // node frames (X red, Y green, Z blue) at the crossings
    if (o.frames && this.nodeR && this.st) {
      const L = this.knotR * 3, pts = [], cols = [];
      const cX = [0.85, 0.25, 0.2], cY = [0.25, 0.65, 0.25], cZ = [0.2, 0.35, 0.85];
      for (const i of this.knotNodes) for (const [axis, cc] of [[0, cX], [1, cY], [2, cZ]]) {
        pts.push(P[3 * i], P[3 * i + 1], P[3 * i + 2], P[3 * i] + L * this.nodeR[9 * i + 3 * axis], P[3 * i + 1] + L * this.nodeR[9 * i + 3 * axis + 1], P[3 * i + 2] + L * this.nodeR[9 * i + 3 * axis + 2]);
        cols.push(...lin(cc), ...lin(cc));
      }
      const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3)); g.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
      this.visible.frames = true; this._setLayer('frames', new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true })));
    } else this._setLayer('frames', null);
  }

  /** solid arrows (shaft + head) from `from` along `vec` (flat arrays, 3 per arrow) */
  _arrows(from, vec, rgb, radius) {
    const n = from.length / 3, g = new THREE.Group(); if (!n) return g;
    const shaft = new THREE.CylinderGeometry(1, 1, 1, 10, 1); shaft.translate(0, 0.5, 0);
    const head = new THREE.ConeGeometry(1, 1, 14, 1); head.translate(0, 0.5, 0);
    const mat = new THREE.MeshStandardMaterial({ color: srgbColor(rgb), roughness: 0.55, metalness: 0 });
    const ms = new THREE.InstancedMesh(shaft, mat, n), mh = new THREE.InstancedMesh(head, mat, n);
    const Y = new THREE.Vector3(0, 1, 0), D = new THREE.Vector3(), Pn = new THREE.Vector3(), q = new THREE.Quaternion(), M4 = new THREE.Matrix4(), S3 = new THREE.Vector3();
    for (let k = 0; k < n; k++) {
      D.set(vec[3 * k], vec[3 * k + 1], vec[3 * k + 2]); const len = D.length() || 1e-12; D.divideScalar(len);
      const r = Math.min(radius, 0.08 * len), hl = Math.min(0.35 * len, 5.5 * r), hr = 2.3 * r;
      q.setFromUnitVectors(Y, D); Pn.set(from[3 * k], from[3 * k + 1], from[3 * k + 2]);
      M4.compose(Pn, q, S3.set(r, len - hl, r)); ms.setMatrixAt(k, M4);
      M4.compose(Pn.clone().addScaledVector(D, len - hl), q, S3.set(hr, hl, hr)); mh.setMatrixAt(k, M4);
    }
    ms.castShadow = mh.castShadow = true;
    g.add(ms, mh); return g;
  }

  /** wide screen-space lines (LineSegments2): positions 6 per segment; colours sRGB per vertex or one colour */
  _fatLines(positions, { rgb = [0.2, 0.2, 0.2], colors = null, width = 1.5, opacity = 1, depthTest = true } = {}) {
    const g = new LineSegmentsGeometry(); g.setPositions(positions);
    if (colors) { const lc = new Float32Array(colors.length); for (let i = 0; i < colors.length; i++) lc[i] = toLinear(colors[i]); g.setColors(lc); }
    const mat = new LineMaterial({ color: colors ? 0xffffff : srgbColor(rgb), linewidth: width, vertexColors: !!colors, transparent: opacity < 1, opacity, depthTest, worldUnits: false });
    mat.resolution.set(this.canvas.clientWidth || 800, this.canvas.clientHeight || 600);
    return new LineSegments2(g, mat);
  }

  _label(text, color, at, size) {
    const key = text + '|' + color;
    let tex = this.labelCache.get(key);
    if (!tex) {
      const S = 128, cv = document.createElement('canvas'); cv.width = 2 * S; cv.height = S;
      const g = cv.getContext('2d');
      g.font = `600 ${0.62 * S}px "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif`; g.textAlign = 'center'; g.textBaseline = 'middle';
      g.lineWidth = 10; g.strokeStyle = this.dark ? 'rgba(20,22,24,0.85)' : 'rgba(255,255,255,0.9)'; g.strokeText(text, S, S / 2);
      g.fillStyle = color; g.fillText(text, S, S / 2);
      tex = new THREE.CanvasTexture(cv); tex.colorSpace = THREE.SRGBColorSpace; this.labelCache.set(key, tex);
    }
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }));
    sp.position.copy(at); sp.scale.set(2 * size, size, 1); sp.renderOrder = 6;
    return sp;
  }

  // --- member diagrams (diagrams.js builds the geometry) ------------------------------------------------------------
  setDiagram(spec) {
    this.diagramSpec = spec;
    if (!spec || (!spec.tris && !spec.lines && !spec.arrows && !spec.polylines)) { this._setLayer('diagram', null); this._setLabels(spec ? spec.labels || [] : []); return; }
    const g = new THREE.Group();
    if (spec.tris && spec.tris.length) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(spec.tris, 3));
      const lc = new Float32Array(spec.triCol.length); for (let i = 0; i < lc.length; i++) lc[i] = toLinear(spec.triCol[i]);
      geo.setAttribute('color', new THREE.BufferAttribute(lc, 3));
      const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: spec.fill ?? 0.35, side: THREE.DoubleSide, depthWrite: false }));
      mesh.renderOrder = 2; g.add(mesh);
    }
    if (spec.lines && spec.lines.length) g.add(this._fatLines(spec.lines, { rgb: this.dark ? [0.78, 0.8, 0.82] : DIAG_OUTLINE, width: 1.1 }));
    if (spec.polylines) for (const pl of spec.polylines) {
      const n = pl.pts.length / 3; if (n < 2) continue;
      const seg = new Float32Array(6 * (n - 1)), col = pl.col ? new Float32Array(6 * (n - 1)) : null;
      for (let i = 0; i < n - 1; i++) { seg.set(pl.pts.subarray(3 * i, 3 * i + 6), 6 * i); if (col) col.set(pl.col.subarray(3 * i, 3 * i + 6), 6 * i); }
      g.add(this._fatLines(seg, { rgb: pl.color || [0.5, 0.5, 0.5], colors: col, width: pl.width || 1.5 }));
    }
    if (spec.arrows && spec.arrows.from.length) g.add(this._arrows(spec.arrows.from, spec.arrows.vec, spec.arrows.color, 0.0022 * this.bboxSize * Math.sqrt(spec.scale || 1)));
    this._setLayer('diagram', g);
    this._setLabels(this.visible.diagram ? spec.labels || [] : []);
  }

  // --- joints (discretisation): cut rectangles + a point per joint coloured by |κ| -----------------------------------
  /** spec: {cuts: Float32Array line segments, points: [[x,y,z]], kappa: [..]} */
  setJoints(spec) {
    this.jointSpec = spec;
    if (!spec || !spec.points.length) { this._setLayer('joints', null); return; }
    const g = new THREE.Group();
    g.add(this._fatLines(spec.cuts, { rgb: this.dark ? [0.85, 0.85, 0.88] : CUT_COLOR, width: 2.2 }));
    const kmax = Math.max(...spec.kappa, 0), r = 0.35 * Math.max(...this.secH);
    const im = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 12, 8), new THREE.MeshStandardMaterial({ roughness: 0.6 }), spec.points.length), M4 = new THREE.Matrix4();
    spec.points.forEach((p, k) => { M4.makeScale(r, r, r).setPosition(p[0], p[1], p[2]); im.setMatrixAt(k, M4); im.setColorAt(k, srgbColor(kmax > 0 ? lutColor('reds', spec.kappa[k] / kmax) : CUT_COLOR)); });
    im.renderOrder = 3; g.add(im);
    this._setLayer('joints', g);
  }

  // --- a lath selection (LocalSH): the other laths dimmed, a tube along each selected lath ---------------------------
  _buildPath() {
    const st = this.st, P = this.dispX, R = this.nodeR;
    if (!st || !P || !this.strandSel.size) { this._setLayer('path', null); return; }
    const g = new THREE.Group(), mat = new THREE.MeshBasicMaterial({ color: this.colFg });
    for (const s of this.strandSel) {
      const i0 = st.nodeOffset[s], n = st.nodeOffset[s + 1] - i0; if (n < 2) continue;
      const h = this.secH[s], r = 0.16 * Math.max(this.secW[s], h), off = 0.5 * h + 0.5 * r;
      const pts = []; for (let i = i0; i < i0 + n; i++) pts.push(new THREE.Vector3(P[3 * i] + off * R[9 * i + 3], P[3 * i + 1] + off * R[9 * i + 4], P[3 * i + 2] + off * R[9 * i + 5]));
      const curve = new THREE.CatmullRomCurve3(pts, this.closed[s], 'centripetal', 0.5);
      const tube = new THREE.Mesh(new THREE.TubeGeometry(curve, Math.max(16, 3 * n), r, 6, this.closed[s]), mat);
      tube.renderOrder = 4; g.add(tube);
    }
    this._setLayer('path', g);
  }

  // --- screen-space value labels (diagrams) -------------------------------------------------------------------------
  _setLabels(list) {
    if (!this.labelsEl) return;
    this.labelsEl.innerHTML = '';
    this.labelItems = list.map(({ p, text }) => {
      const el = document.createElement('div'); el.className = 'dlabel';
      el.innerHTML = `<i></i><span>${text}</span>`;
      this.labelsEl.appendChild(el);
      return { p: new THREE.Vector3(p[0], p[1], p[2]), el };
    });
    this.needsRender = true;
  }
  _placeLabels() {
    if (!this.labelItems.length) return;
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight, v = new THREE.Vector3();
    for (const it of this.labelItems) {
      v.copy(it.p).project(this.camera);
      const vis = v.z > -1 && v.z < 1 && Math.abs(v.x) <= 1.02 && Math.abs(v.y) <= 1.02;
      it.el.style.display = vis ? '' : 'none';
      if (vis) it.el.style.transform = `translate(${((v.x + 1) / 2 * w).toFixed(1)}px, ${((1 - v.y) / 2 * h).toFixed(1)}px)`;
    }
  }

  // --- picking ------------------------------------------------------------------------------------------------------
  /** → {kind: 'lath', seg, strand, point} | {kind: 'knot', c, point} | {kind: 'mesh', point} | null */
  pick(clientX, clientY) {
    const r = this.canvas.getBoundingClientRect();
    const nd = new THREE.Vector2(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(nd, this.camera);
    const targets = [];
    if (this.layers.knots && this.layers.knots.visible) targets.push(this.layers.knots);
    if (this.layers.laths && this.layers.laths.visible) targets.push(this.layers.laths);
    if (this.layers.mesh && this.layers.mesh.visible) targets.push(this.layers.mesh);
    const hits = this.raycaster.intersectObjects(targets, false);
    if (!hits.length) return null;
    // knot spheres sit inside the laths: prefer a knot hit within a lath thickness behind the first hit
    const h0 = hits[0], kh = hits.find(h => h.object.userData.kind === 'knot');
    const h = kh && kh.distance <= h0.distance + 2.5 * this.knotR ? kh : h0;
    const kind = h.object.userData.kind;
    if (kind === 'knot') return { kind: 'knot', c: h.instanceId, point: h.point };
    if (kind === 'lath') { const seg = Math.floor(h.faceIndex / 8); return { kind: 'lath', seg, strand: this.st.segStrand[seg], point: h.point }; }
    return { kind: 'mesh', point: h.point };
  }

  /** frame id [s, k] of the crossing on `strand` nearest to `point` (a click on a lath) */
  nearestFrame(strand, point) {
    const st = this.st, topo = this.topo, P = this.dispX;
    let best = null, bd = Infinity;
    const edges = topo.strands[strand].edges;
    for (let k = 0; k < edges.length; k++) {
      const nid = st.nodeFrameIds.get(fkey(strand, k)); if (nid === undefined) continue;
      const d = Math.hypot(P[3 * nid] - point.x, P[3 * nid + 1] - point.y, P[3 * nid + 2] - point.z);
      if (d < bd) { bd = d; best = [strand, k]; }
    }
    return best;
  }

  screenshotDataURL() { this.renderer.render(this.scene, this.camera); return this.canvas.toDataURL('image/png'); }
}

function slerp(a, b, t) {
  const d = Math.min(1, Math.max(-1, a.dot(b)));
  if (d > 0.9995) return a.clone().lerp(b, t).normalize();
  const th = Math.acos(d) * t, rel = b.clone().addScaledVector(a, -d).normalize();
  return a.clone().multiplyScalar(Math.cos(th)).addScaledVector(rel, Math.sin(th));
}
function disposeObject(o) {
  o.traverse((c) => {
    if (c.geometry) c.geometry.dispose();
    if (c.material && !c.userData.sharedMaterial) (Array.isArray(c.material) ? c.material : [c.material]).forEach(m => m.dispose());
  });
}
