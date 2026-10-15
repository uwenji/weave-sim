"""Generate example meshes: a regular-lattice flat sheet and a spherical-cap dome (mm)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "meshes"


def tri_lattice(nx: int, ny: int, a: float) -> tuple[np.ndarray, np.ndarray]:
    """Regular triangular lattice with edge length a, nx x ny vertices (parallelogram)."""
    V = []
    for j in range(ny):
        for i in range(nx):
            V.append([a * (i + 0.5 * j), a * j * np.sqrt(3) / 2, 0.0])
    V = np.array(V)
    F = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            v00 = j * nx + i
            v10 = v00 + 1
            v01 = v00 + nx
            v11 = v01 + 1
            F.append([v00, v10, v11])
            F.append([v00, v11, v01])
    return V, np.array(F, dtype=np.int64)


def sheet(nx=9, ny=8, a=300.0):
    return tri_lattice(nx, ny, a)


def dome(radius=3000.0, cap_height=1200.0, spacing=350.0):
    """Spherical cap triangulated from a hex-ish lattice on the base disc projected onto the sphere."""
    r_base = np.sqrt(cap_height * (2 * radius - cap_height))
    n = int(np.ceil(2 * r_base / spacing)) + 3
    V, F = tri_lattice(n, n, spacing)
    V[:, :2] -= V[:, :2].mean(axis=0)
    # keep vertices within the disc, re-triangulate the kept set
    keep = np.linalg.norm(V[:, :2], axis=1) <= r_base * 1.0001
    new_idx = -np.ones(len(V), dtype=int)
    new_idx[keep] = np.arange(keep.sum())
    V = V[keep]
    F = new_idx[F]
    F = F[(F >= 0).all(axis=1)]
    # lift onto sphere of given radius centred below the plane
    zc = -(radius - cap_height)
    rho2 = (V[:, 0] ** 2 + V[:, 1] ** 2)
    V[:, 2] = zc + np.sqrt(np.maximum(radius**2 - rho2, 0.0))
    return V, F


def write_stl(path: Path, V: np.ndarray, F: np.ndarray):
    import trimesh

    trimesh.Trimesh(V, F, process=False).export(str(path))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    V, F = sheet()
    write_stl(OUT / "sheet.stl", V, F)
    print("sheet", V.shape, F.shape)
    V, F = dome()
    write_stl(OUT / "dome.stl", V, F)
    print("dome", V.shape, F.shape)
