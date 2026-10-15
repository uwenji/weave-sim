// Navigation cube (CAD style): a chamfered cube with 6 labelled faces, 12 edge strips and 8 corner facets that turns
// with the main camera. Clicking a facet turns the view to look along that facet's normal; dragging the cube orbits
// like dragging the model. Draws into its own small transparent WebGL canvas; a small xyz triad sits bottom-left.
// Z is up: FRONT faces −y, RIGHT faces +x, TOP faces +z.
import * as THREE from 'three';

const CHAMFER = 0.30, S_ = 1 - CHAMFER;
const FACES = [
  { id: 'right', n: [1, 0, 0], up: [0, 0, 1], label: 'RIGHT' }, { id: 'left', n: [-1, 0, 0], up: [0, 0, 1], label: 'LEFT' },
  { id: 'back', n: [0, 1, 0], up: [0, 0, 1], label: 'BACK' }, { id: 'front', n: [0, -1, 0], up: [0, 0, 1], label: 'FRONT' },
  { id: 'top', n: [0, 0, 1], up: [0, 1, 0], label: 'TOP' }, { id: 'bottom', n: [0, 0, -1], up: [0, 1, 0], label: 'BOTTOM' },
];
const AXIS_COLORS = [0xd0392b, 0x3a9b45, 0x2d6fd1];
const cssVar = (name, fallback) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;

export class NavCube {
  /** opts: { onPick(dirArray, facet), onDrag(dx, dy) } */
  constructor(canvas, opts = {}) {
    this.canvas = canvas; this.opts = opts;
    this.renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, powerPreference: 'low-power' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setClearColor(0x000000, 0);
    this.scene = new THREE.Scene();
    this.L = 1.62;                                          // corner facets reach radius 1.41, so the cube always fits
    this.camera = new THREE.OrthographicCamera(-this.L, this.L, this.L, -this.L, 0.1, 30); this.camera.up.set(0, 0, 1);
    this.raycaster = new THREE.Raycaster();
    this.facets = [];                                       // { mesh, dir, kind, name, baseColor }
    this.hover = null;
    this._build();
    this._bind();
    this.applyTheme();
    this.resize();
  }

  resize() {
    const w = this.canvas.clientWidth || 108, h = this.canvas.clientHeight || 108;
    this.renderer.setSize(w, h, false);
  }

  // --- geometry ---------------------------------------------------------------------------------------------------
  _build() {
    const s = S_, group = new THREE.Group(); this.group = group; this.scene.add(group);
    const outline = [];
    const addFacet = (verts, dir, kind, name, material) => {
      const geo = new THREE.BufferGeometry();
      const flat = []; for (const v of verts) flat.push(...v);
      geo.setAttribute('position', new THREE.BufferAttribute(Float32Array.from(flat), 3));
      const idx = []; for (let i = 1; i < verts.length - 1; i++) idx.push(0, i, i + 1);
      geo.setIndex(idx); geo.computeVertexNormals();
      // outward orientation
      const n = new THREE.Vector3().fromArray(geo.attributes.normal.array, 0), cen = new THREE.Vector3();
      for (const v of verts) cen.add(new THREE.Vector3(...v)); cen.divideScalar(verts.length);
      if (n.dot(cen) < 0) { geo.setIndex(idx.reverse()); geo.computeVertexNormals(); }
      const mesh = new THREE.Mesh(geo, material); mesh.userData.facet = this.facets.length; group.add(mesh);
      for (let i = 0; i < verts.length; i++) outline.push(...verts[i], ...verts[(i + 1) % verts.length]);
      this.facets.push({ mesh, dir: new THREE.Vector3(...dir).normalize(), kind, name });
      return mesh;
    };
    // faces: squares of half-size s, textured with the label (built in applyTheme)
    this.faceMeshes = [];
    for (const f of FACES) {
      const z = new THREE.Vector3(...f.n), y = new THREE.Vector3(...f.up), x = new THREE.Vector3().crossVectors(y, z);
      const corners = [[-s, -s], [s, -s], [s, s], [-s, s]].map(([u, v]) => z.clone().addScaledVector(x, u).addScaledVector(y, v).toArray());
      const mat = new THREE.MeshBasicMaterial({ color: 0xffffff });
      const mesh = addFacet(corners, f.n, 'face', f.label.toLowerCase(), mat);
      // uv for the label texture: u along x, v along y
      mesh.geometry.setAttribute('uv', new THREE.BufferAttribute(Float32Array.from([0, 0, 1, 0, 1, 1, 0, 1]), 2));
      this.faceMeshes.push({ mesh, label: f.label });
    }
    // edge strips between every pair of perpendicular faces
    const axes = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
    for (let i = 0; i < 6; i++) for (let j = i + 1; j < 6; j++) {
      const A = new THREE.Vector3(...axes[i]), B = new THREE.Vector3(...axes[j]); if (Math.abs(A.dot(B)) > 0.5) continue;
      const E = new THREE.Vector3().crossVectors(A, B).normalize();
      const p = (a, b, e) => A.clone().multiplyScalar(a).addScaledVector(B, b).addScaledVector(E, e).toArray();
      const verts = [p(1, s, -s), p(1, s, s), p(s, 1, s), p(s, 1, -s)];
      addFacet(verts, A.clone().add(B).toArray(), 'edge', `${faceName(axes[i])}-${faceName(axes[j])}`, new THREE.MeshBasicMaterial({ color: 0xffffff }));
    }
    // corner triangles
    for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) {
      const verts = [[sx, sy * s, sz * s], [sx * s, sy, sz * s], [sx * s, sy * s, sz]];
      addFacet(verts, [sx, sy, sz], 'corner', `${faceName([sx, 0, 0])}-${faceName([0, sy, 0])}-${faceName([0, 0, sz])}`, new THREE.MeshBasicMaterial({ color: 0xffffff }));
    }
    const lg = new THREE.BufferGeometry(); lg.setAttribute('position', new THREE.BufferAttribute(Float32Array.from(outline), 3));
    this.outline = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: 0x888888, transparent: true, opacity: 0.9 }));
    group.add(this.outline);
    // xyz triad: world-aligned lines at a fixed spot of the little viewport (positioned each frame in camera space)
    this.triad = new THREE.Group();
    AXIS_COLORS.forEach((col, k) => {
      const a = [0, 0, 0], b = [0, 0, 0]; b[k] = 0.5;
      const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(Float32Array.from([...a, ...b]), 3));
      this.triad.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: col })));
    });
    this.scene.add(this.triad);
  }

  // --- theme ------------------------------------------------------------------------------------------------------
  applyTheme() {
    const dark = cssVar('--scheme', 'light') === 'dark';
    const face = new THREE.Color(cssVar(dark ? '--panel-2' : '--panel', dark ? '#2e3135' : '#fbfbfa'));
    const fg = new THREE.Color(cssVar('--fg', '#212324'));
    this.colFace = face; this.colAccent = new THREE.Color(cssVar('--accent', '#2973c7'));
    this.outline.material.color.copy(new THREE.Color(cssVar('--mesh-edge', '#b8b8b5')));
    const shade = (k) => face.clone().lerp(fg, k);
    for (const f of this.facets) {
      f.base = f.kind === 'face' ? face.clone() : f.kind === 'edge' ? shade(dark ? 0.08 : 0.05) : shade(dark ? 0.16 : 0.10);
      f.mesh.material.color.copy(f.base);
    }
    // face labels: the texture carries the fill, the material colour stays white (a tinted material would square the fill)
    const text = '#' + fg.clone().lerp(face, 0.12).getHexString(), hoverFill = '#' + face.clone().lerp(this.colAccent, 0.45).getHexString();
    for (const fm of this.faceMeshes) {
      for (const t of [fm.tex, fm.texHover]) t?.dispose();
      fm.tex = labelTexture(fm.label, '#' + face.getHexString(), text); fm.texHover = labelTexture(fm.label, hoverFill, text);
      fm.mesh.material.color.set(0xffffff); fm.mesh.material.map = fm.tex; fm.mesh.material.needsUpdate = true;
    }
    this._applyHover();
  }

  _applyHover() {
    for (const f of this.facets) {
      if (f.kind === 'face') { const fm = this.faceMeshes.find(x => x.mesh === f.mesh); f.mesh.material.map = f === this.hover ? fm.texHover : fm.tex; }
      else f.mesh.material.color.copy(f === this.hover ? f.base.clone().lerp(this.colAccent, 0.45) : f.base);
    }
    this.canvas.style.cursor = this.hover ? 'pointer' : 'grab';
    this.canvas.title = this.hover ? `look at ${this.hover.name.replace(/-/g, ' · ')}` : 'click a face, edge or corner to look from there · drag to orbit';
  }

  // --- interaction --------------------------------------------------------------------------------------------------
  _bind() {
    const cv = this.canvas; let down = null, dragged = false, last = null;
    cv.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; last = down; dragged = false; cv.setPointerCapture(e.pointerId); e.preventDefault(); });
    cv.addEventListener('pointermove', (e) => {
      if (down) {
        if (!dragged && Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) { dragged = true; this.hover = null; this._applyHover(); cv.style.cursor = 'grabbing'; }
        if (dragged) { this.opts.onDrag?.(e.clientX - last[0], e.clientY - last[1]); last = [e.clientX, e.clientY]; return; }
      }
      const f = this._facetAt(e.clientX, e.clientY);
      if (f !== this.hover) { this.hover = f; this._applyHover(); }
    });
    const up = (e) => {
      if (!down) return;
      const wasDrag = dragged; down = null; dragged = false;
      try { cv.releasePointerCapture(e.pointerId); } catch (err) { /* ignore */ }
      if (wasDrag) { this._applyHover(); return; }
      const f = this._facetAt(e.clientX, e.clientY);
      if (f) this.opts.onPick?.(f.dir.toArray(), f);
    };
    cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
    cv.addEventListener('pointerleave', () => { if (!down && this.hover) { this.hover = null; this._applyHover(); } });
    cv.addEventListener('dblclick', (e) => e.preventDefault());
  }

  _facetAt(clientX, clientY) {
    const r = this.canvas.getBoundingClientRect();
    const nd = new THREE.Vector2(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(nd, this.camera);
    const hits = this.raycaster.intersectObjects(this.facets.map(f => f.mesh), false);
    return hits.length ? this.facets[hits[0].object.userData.facet] : null;
  }

  // --- per frame ----------------------------------------------------------------------------------------------------
  /** mirror the main camera: same orientation, the cube seen from the same direction */
  render(mainCamera, target) {
    const dir = mainCamera.position.clone().sub(target); if (dir.lengthSq() < 1e-12) dir.set(0, -1, 0.5);
    dir.normalize();
    this.camera.position.copy(dir).multiplyScalar(8);
    this.camera.quaternion.copy(mainCamera.quaternion);
    this.camera.updateMatrixWorld();
    // triad pinned to the bottom-left of the little viewport, in front of the cube
    this.triad.position.copy(this.camera.localToWorld(new THREE.Vector3(-this.L + 0.42, -this.L + 0.42, -4.5)));
    this.renderer.render(this.scene, this.camera);
  }

  dispose() { this.renderer.dispose(); }
}

function faceName(n) { return n[0] > 0 ? 'right' : n[0] < 0 ? 'left' : n[1] > 0 ? 'back' : n[1] < 0 ? 'front' : n[2] > 0 ? 'top' : 'bottom'; }

function labelTexture(text, fill, color) {
  const S = 192, cv = document.createElement('canvas'); cv.width = S; cv.height = S;
  const ctx = cv.getContext('2d');
  ctx.fillStyle = fill; ctx.fillRect(0, 0, S, S);
  const font = (px) => `600 ${px}px "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif`;
  let px = 40; ctx.font = font(px);
  px = Math.min(0.42 * S, Math.floor(px * 0.74 * S / ctx.measureText(text).width)); ctx.font = font(px);   // text spans ~74 % of the face
  ctx.fillStyle = color; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(text, S / 2, S / 2 + px * 0.05);
  const tex = new THREE.CanvasTexture(cv); tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 4; tex.minFilter = THREE.LinearMipmapLinearFilter;
  return tex;
}
