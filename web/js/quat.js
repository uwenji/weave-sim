// Quaternion and frame helpers in f64 (port of weavesim/solver/quat.py). Quaternion layout (w, x, y, z);
// a frame is 9 numbers [X | Y | Z] (the columns of R: X = width, Y = normal, Z = tangent).

export function qnormalize(q, o = 0) {
  let n = Math.sqrt(q[o] * q[o] + q[o + 1] * q[o + 1] + q[o + 2] * q[o + 2] + q[o + 3] * q[o + 3]);
  if (n === 0) n = 1.0;
  q[o] /= n; q[o + 1] /= n; q[o + 2] /= n; q[o + 3] /= n;
}
/** a ⊗ b → out (arrays of 4) */
export function qmul(a, b, out = new Float64Array(4)) {
  const aw = a[0], ax = a[1], ay = a[2], az = a[3], bw = b[0], bx = b[1], by = b[2], bz = b[3];
  out[0] = aw * bw - ax * bx - ay * by - az * bz;
  out[1] = aw * bx + ax * bw + ay * bz - az * by;
  out[2] = aw * by - ax * bz + ay * bw + az * bx;
  out[3] = aw * bz + ax * by - ay * bx + az * bw;
  return out;
}
/** frame of quaternion Q[o..o+3] (normalised first) → R[9] = [X | Y | Z] */
export function toFrame(Q, o, R = new Float64Array(9)) {
  let w = Q[o], x = Q[o + 1], y = Q[o + 2], z = Q[o + 3];
  let n = Math.sqrt(w * w + x * x + y * y + z * z); if (n === 0) n = 1.0;
  w /= n; x /= n; y /= n; z /= n;
  R[0] = 1 - 2 * (y * y + z * z); R[1] = 2 * (x * y + z * w); R[2] = 2 * (x * z - y * w);
  R[3] = 2 * (x * y - z * w); R[4] = 1 - 2 * (x * x + z * z); R[5] = 2 * (y * z + x * w);
  R[6] = 2 * (x * z + y * w); R[7] = 2 * (y * z - x * w); R[8] = 1 - 2 * (x * x + y * y);
  return R;
}
/** frame [X|Y|Z] → quaternion (Shepperd, the setup version: strict comparisons) */
export function fromFrame(F, out = new Float64Array(4)) {
  // matrix element m[r][c] = F[3c + r]
  const m00 = F[0], m10 = F[1], m20 = F[2], m01 = F[3], m11 = F[4], m21 = F[5], m02 = F[6], m12 = F[7], m22 = F[8];
  const t = m00 + m11 + m22;
  if (t > 0) { const s = Math.sqrt(t + 1.0) * 2; out[0] = 0.25 * s; out[1] = (m21 - m12) / s; out[2] = (m02 - m20) / s; out[3] = (m10 - m01) / s; }
  else if (m00 > m11 && m00 > m22) { const s = Math.sqrt(1.0 + m00 - m11 - m22) * 2; out[0] = (m21 - m12) / s; out[1] = 0.25 * s; out[2] = (m01 + m10) / s; out[3] = (m02 + m20) / s; }
  else if (m11 > m22) { const s = Math.sqrt(1.0 + m11 - m00 - m22) * 2; out[0] = (m02 - m20) / s; out[1] = (m01 + m10) / s; out[2] = 0.25 * s; out[3] = (m12 + m21) / s; }
  else { const s = Math.sqrt(1.0 + m22 - m00 - m11) * 2; out[0] = (m10 - m01) / s; out[1] = (m02 + m20) / s; out[2] = (m12 + m21) / s; out[3] = 0.25 * s; }
  qnormalize(out);
  return out;
}
/** the vectorised variant used every iteration (non-strict comparisons, clamped roots) */
export function fromFrameFast(F, out = new Float64Array(4)) {
  const m00 = F[0], m10 = F[1], m20 = F[2], m01 = F[3], m11 = F[4], m21 = F[5], m02 = F[6], m12 = F[7], m22 = F[8];
  const t = m00 + m11 + m22;
  if (t > 0) { const s = Math.sqrt(Math.max(t + 1.0, 0.0)) * 2; out[0] = 0.25 * s; out[1] = (m21 - m12) / s; out[2] = (m02 - m20) / s; out[3] = (m10 - m01) / s; }
  else if (m00 >= m11 && m00 >= m22) { const s = Math.sqrt(Math.max(1.0 + m00 - m11 - m22, 0.0)) * 2; out[0] = (m21 - m12) / s; out[1] = 0.25 * s; out[2] = (m01 + m10) / s; out[3] = (m02 + m20) / s; }
  else if (m11 >= m22) { const s = Math.sqrt(Math.max(1.0 + m11 - m00 - m22, 0.0)) * 2; out[0] = (m02 - m20) / s; out[1] = (m01 + m10) / s; out[2] = 0.25 * s; out[3] = (m12 + m21) / s; }
  else { const s = Math.sqrt(Math.max(1.0 + m22 - m00 - m11, 0.0)) * 2; out[0] = (m10 - m01) / s; out[1] = (m02 + m20) / s; out[2] = (m12 + m21) / s; out[3] = 0.25 * s; }
  qnormalize(out);
  return out;
}
/** rotation vector → unit quaternion */
export function qexp(tx, ty, tz, out = new Float64Array(4)) {
  const a = Math.sqrt(tx * tx + ty * ty + tz * tz), half = 0.5 * a;
  const s = a > 1e-12 ? Math.sin(half) / a : 0.5;
  out[0] = Math.cos(half); out[1] = tx * s; out[2] = ty * s; out[3] = tz * s;
  return out;
}
/** minimal rotation (row-major 3×3) taking unit a onto unit b: I + K + (1 − c)/|v|² K² */
export function rotBetween(ax, ay, az, bx, by, bz, R = new Float64Array(9)) {
  const vx = ay * bz - az * by, vy = az * bx - ax * bz, vz = ax * by - ay * bx;
  const c = ax * bx + ay * by + az * bz, s2 = vx * vx + vy * vy + vz * vz;
  const fac = s2 > 1e-24 ? (1 - c) / Math.max(s2, 1e-24) : 0.5;
  // K = [[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]]
  const K = [0, -vz, vy, vz, 0, -vx, -vy, vx, 0];
  for (let i = 0; i < 3; i++) for (let k = 0; k < 3; k++) {
    let kk = 0; for (let j = 0; j < 3; j++) kk += K[3 * i + j] * K[3 * j + k];
    R[3 * i + k] = (i === k ? 1 : 0) + K[3 * i + k] + fac * kk;
  }
  return R;
}
