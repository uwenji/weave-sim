// Triangle mesh with the adjacency the kagome tracer needs (port of weavesim/mesh.py).
// Vertex and edge numbering follow the Python program exactly (weld = lexicographic order of the quantised
// coordinates, edges = lexicographic order of the sorted vertex pairs), so knot ids, strand ids and frame ids in a
// scene.json written by the Python viewer address the same crossings here.

export class Mesh {
  /** V: Float64Array(3n), F: Int32Array(3m) (consistently oriented triangles) */
  static fromArrays(V, F, path = '') {
    const m = new Mesh();
    m.V = Float64Array.from(V); m.F = Int32Array.from(F); m.path = path;
    m.nV = m.V.length / 3; m.nF = m.F.length / 3;
    edgeTopology(m);
    m.faceNormals = faceNormals(m.V, m.F);
    m.vertexNormals = vertexNormals(m.V, m.F, m.faceNormals);
    m.edgeNormals = edgeNormals(m.EF, m.faceNormals, m.nE);
    m.isBoundaryEdge = new Uint8Array(m.nE);
    for (let e = 0; e < m.nE; e++) m.isBoundaryEdge[e] = (m.EF[2 * e] < 0 || m.EF[2 * e + 1] < 0) ? 1 : 0;
    m.edgeMidpoints = new Float64Array(3 * m.nE);
    for (let e = 0; e < m.nE; e++) for (let c = 0; c < 3; c++) m.edgeMidpoints[3 * e + c] = 0.5 * (m.V[3 * m.E[2 * e] + c] + m.V[3 * m.E[2 * e + 1] + c]);
    return m;
  }

  /** parse a file's bytes (STL binary/ASCII or OBJ), weld duplicate vertices, scale */
  static fromBytes(buf, name = 'mesh.stl', scale = 1.0) {
    const lower = name.toLowerCase();
    let tri;
    if (lower.endsWith('.obj')) tri = parseOBJ(new TextDecoder().decode(buf));
    else tri = parseSTL(buf);
    const V = tri.V; for (let i = 0; i < V.length; i++) V[i] *= scale;
    const w = weld(V, tri.F);
    return Mesh.fromArrays(w.V, w.F, name);
  }

  bbox() {
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < this.nV; i++) for (let c = 0; c < 3; c++) { const v = this.V[3 * i + c]; if (v < lo[c]) lo[c] = v; if (v > hi[c]) hi[c] = v; }
    return { lo, hi, size: Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) };
  }
  otherFace(edge, face) { const a = this.EF[2 * edge], b = this.EF[2 * edge + 1]; return a === face ? b : a; }
  edgeHas(e, v) { return this.E[2 * e] === v || this.E[2 * e + 1] === v; }
  /** edge id of `face` that contains `vertex` and is not `excludeEdge` */
  faceEdgeContaining(face, vertex, excludeEdge) {
    for (let i = 0; i < 3; i++) {
      const e = this.FE[3 * face + i];
      if (e === excludeEdge) continue;
      if (this.edgeHas(e, vertex)) return e;
    }
    throw new Error(`face ${face} has no edge containing vertex ${vertex} other than edge ${excludeEdge}`);
  }
  area() {
    let a = 0; const V = this.V, F = this.F;
    for (let f = 0; f < this.nF; f++) {
      const i = 3 * F[3 * f], j = 3 * F[3 * f + 1], k = 3 * F[3 * f + 2];
      const ux = V[j] - V[i], uy = V[j + 1] - V[i + 1], uz = V[j + 2] - V[i + 2], vx = V[k] - V[i], vy = V[k + 1] - V[i + 1], vz = V[k + 2] - V[i + 2];
      a += 0.5 * Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
    }
    return a;
  }
}

// --- parsing ------------------------------------------------------------------------------------------------------
export function parseSTL(buf) {
  const u8 = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  if (u8.byteLength >= 84) {
    const n = dv.getUint32(80, true);
    if (84 + 50 * n === u8.byteLength) {
      const V = new Float64Array(9 * n), F = new Int32Array(3 * n);
      for (let t = 0; t < n; t++) {
        const o = 84 + 50 * t + 12;
        for (let k = 0; k < 9; k++) V[9 * t + k] = dv.getFloat32(o + 4 * k, true);
        F[3 * t] = 3 * t; F[3 * t + 1] = 3 * t + 1; F[3 * t + 2] = 3 * t + 2;
      }
      return { V, F };
    }
  }
  const text = new TextDecoder().decode(u8);
  const re = /vertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)/g;
  const vals = []; let m;
  while ((m = re.exec(text))) vals.push(+m[1], +m[2], +m[3]);
  const nt = Math.floor(vals.length / 9);
  const V = Float64Array.from(vals.slice(0, 9 * nt)), F = new Int32Array(3 * nt);
  for (let i = 0; i < 3 * nt; i++) F[i] = i;
  if (!nt) throw new Error('no triangles found (binary or ASCII STL expected)');
  return { V, F };
}

export function parseOBJ(text) {
  const V = [], F = [];
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (line.startsWith('v ')) { const p = line.split(/\s+/); V.push(+p[1], +p[2], +p[3]); }
    else if (line.startsWith('f ')) {
      const nv = V.length / 3;
      const idx = line.split(/\s+/).slice(1).map(tok => { const i = parseInt(tok.split('/')[0], 10); return i < 0 ? nv + i : i - 1; });
      for (let k = 1; k + 1 < idx.length; k++) F.push(idx[0], idx[k], idx[k + 1]);   // fan, as igl::polygon_mesh_to_triangle_mesh
    }
  }
  if (!F.length) throw new Error('no faces in the OBJ file');
  return { V: Float64Array.from(V), F: Int32Array.from(F) };
}

/** np.round: round half to even */
function roundHalfEven(x) { const r = Math.round(x); return Math.abs(x % 1) === 0.5 ? 2 * Math.round(x / 2) : r; }

/** weld duplicated vertices (STL): quantise at tol × model size, unique rows in lexicographic order (np.unique axis=0,
 *  first occurrence kept), drop the faces that collapse */
export function weld(V, F, tol = 1e-9) {
  const n = V.length / 3;
  if (!n) return { V, F };
  let ptp = 0;
  for (let c = 0; c < 3; c++) { let lo = Infinity, hi = -Infinity; for (let i = 0; i < n; i++) { const v = V[3 * i + c]; if (v < lo) lo = v; if (v > hi) hi = v; } ptp = Math.max(ptp, hi - lo); }
  const scale = Math.max(ptp, 1.0), q = scale * tol;
  const key = new Float64Array(3 * n);
  for (let i = 0; i < 3 * n; i++) key[i] = roundHalfEven(V[i] / q);
  const order = Array.from({ length: n }, (_, i) => i);
  order.sort((a, b) => (key[3 * a] - key[3 * b]) || (key[3 * a + 1] - key[3 * b + 1]) || (key[3 * a + 2] - key[3 * b + 2]) || (a - b));
  const inv = new Int32Array(n), first = [];
  let u = -1;
  for (let r = 0; r < n; r++) {
    const i = order[r];
    if (r === 0 || key[3 * i] !== key[3 * order[r - 1]] || key[3 * i + 1] !== key[3 * order[r - 1] + 1] || key[3 * i + 2] !== key[3 * order[r - 1] + 2]) { u++; first.push(i); }
    inv[i] = u;
  }
  const Vw = new Float64Array(3 * first.length);
  first.forEach((i, k) => { Vw[3 * k] = V[3 * i]; Vw[3 * k + 1] = V[3 * i + 1]; Vw[3 * k + 2] = V[3 * i + 2]; });
  const Fw = [];
  for (let f = 0; f < F.length / 3; f++) {
    const a = inv[F[3 * f]], b = inv[F[3 * f + 1]], c = inv[F[3 * f + 2]];
    if (a !== b && b !== c && a !== c) Fw.push(a, b, c);
  }
  return { V: Vw, F: Int32Array.from(Fw) };
}

/** edge i of a face is opposite local vertex i: (F[(i+1)%3], F[(i+2)%3]); E sorted unique pairs; EF in the
 *  same fill order as weavesim.mesh._edge_topology */
function edgeTopology(m) {
  const F = m.F, nF = m.nF;
  const rows = new Array(3 * nF);
  for (let i = 0; i < 3; i++) for (let f = 0; f < nF; f++) {
    const a = F[3 * f + (i + 1) % 3], b = F[3 * f + (i + 2) % 3];
    rows[i * nF + f] = a < b ? [a, b] : [b, a];
  }
  const order = Array.from({ length: 3 * nF }, (_, r) => r);
  order.sort((r, s) => (rows[r][0] - rows[s][0]) || (rows[r][1] - rows[s][1]) || (r - s));
  const inv = new Int32Array(3 * nF), E = [];
  let u = -1;
  for (let k = 0; k < order.length; k++) {
    const r = order[k];
    if (k === 0 || rows[r][0] !== rows[order[k - 1]][0] || rows[r][1] !== rows[order[k - 1]][1]) { u++; E.push(rows[r][0], rows[r][1]); }
    inv[r] = u;
  }
  m.nE = E.length / 2; m.E = Int32Array.from(E);
  m.FE = new Int32Array(3 * nF);
  for (let i = 0; i < 3; i++) for (let f = 0; f < nF; f++) m.FE[3 * f + i] = inv[i * nF + f];
  m.EF = new Int32Array(2 * m.nE).fill(-1);
  for (let r = 0; r < 3 * nF; r++) {
    const e = inv[r], f = r % nF;
    if (m.EF[2 * e] < 0) m.EF[2 * e] = f;
    else if (m.EF[2 * e + 1] < 0) m.EF[2 * e + 1] = f;
    else throw new Error(`non-manifold edge (${m.E[2 * e]}, ${m.E[2 * e + 1]}) shared by more than two faces`);
  }
}

function faceNormals(V, F) {
  const nF = F.length / 3, out = new Float64Array(3 * nF);
  for (let f = 0; f < nF; f++) {
    const i = 3 * F[3 * f], j = 3 * F[3 * f + 1], k = 3 * F[3 * f + 2];
    const ux = V[j] - V[i], uy = V[j + 1] - V[i + 1], uz = V[j + 2] - V[i + 2], vx = V[k] - V[i], vy = V[k + 1] - V[i + 1], vz = V[k + 2] - V[i + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    let l = Math.sqrt(nx * nx + ny * ny + nz * nz); if (l === 0) l = 1;
    out[3 * f] = nx / l; out[3 * f + 1] = ny / l; out[3 * f + 2] = nz / l;
  }
  return out;
}
function vertexNormals(V, F, fn) {
  const n = V.length / 3, nF = F.length / 3, out = new Float64Array(3 * n);
  for (let i = 0; i < 3; i++) for (let f = 0; f < nF; f++) { const v = F[3 * f + i]; for (let c = 0; c < 3; c++) out[3 * v + c] += fn[3 * f + c]; }
  for (let v = 0; v < n; v++) { let l = Math.sqrt(out[3 * v] * out[3 * v] + out[3 * v + 1] * out[3 * v + 1] + out[3 * v + 2] * out[3 * v + 2]); if (l === 0) l = 1; for (let c = 0; c < 3; c++) out[3 * v + c] /= l; }
  return out;
}
function edgeNormals(EF, fn, nE) {
  const out = new Float64Array(3 * nE);
  for (let side = 0; side < 2; side++) for (let e = 0; e < nE; e++) { const f = EF[2 * e + side]; if (f >= 0) for (let c = 0; c < 3; c++) out[3 * e + c] += fn[3 * f + c]; }
  for (let e = 0; e < nE; e++) { let l = Math.sqrt(out[3 * e] * out[3 * e] + out[3 * e + 1] * out[3 * e + 1] + out[3 * e + 2] * out[3 * e + 2]); if (l === 0) l = 1; for (let c = 0; c < 3; c++) out[3 * e + c] /= l; }
  return out;
}
