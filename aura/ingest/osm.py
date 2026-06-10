"""Pull Nanshan OSM street graph and derive station-candidate centroids.

Stage 1: 80 m DBSCAN dedupe on segment midpoints; ~5000 candidates target.
Heavy imports are deferred to keep `import aura.ingest.osm` cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass
class StreetCentroids:
    xy_m: np.ndarray            # (N, 2) projected meters
    crs: str = "EPSG:32650"
    n_segments_total: int = 0


def fetch_street_graph(bbox_wgs84: Tuple[float, float, float, float],
                       network_type: str = "drive",
                       cache_path: str | None = None):
    """Return an osmnx MultiDiGraph for the bbox.

    `bbox_wgs84` = (min_lon, min_lat, max_lon, max_lat).
    """
    import osmnx as ox  # heavy
    from pathlib import Path as _P
    if cache_path and _P(cache_path).exists():
        return ox.load_graphml(cache_path)
    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    G = ox.graph_from_bbox(
        bbox=(max_lat, min_lat, max_lon, min_lon),
        network_type=network_type,
        retain_all=False,
        truncate_by_edge=True,
    )
    if cache_path:
        ox.save_graphml(G, cache_path)
    return G


def segment_midpoints(graph, target_crs: str = "EPSG:32650") -> np.ndarray:
    """Collect edge midpoints in the projected CRS as an (N, 2) array."""
    import osmnx as ox
    G_proj = ox.project_graph(graph, to_crs=target_crs)
    mids: list[tuple[float, float]] = []
    for u, v, _, data in G_proj.edges(keys=True, data=True):
        geom = data.get("geometry")
        if geom is not None:
            try:
                xs, ys = geom.xy
                xs = list(xs); ys = list(ys)
                xm = sum(xs) / len(xs)
                ym = sum(ys) / len(ys)
            except Exception:
                continue
        else:
            ux, uy = G_proj.nodes[u]["x"], G_proj.nodes[u]["y"]
            vx, vy = G_proj.nodes[v]["x"], G_proj.nodes[v]["y"]
            xm = 0.5 * (ux + vx)
            ym = 0.5 * (uy + vy)
        mids.append((xm, ym))
    return np.asarray(mids, dtype=np.float64)


def dbscan_dedupe(points: np.ndarray, radius_m: float = 80.0) -> np.ndarray:
    """Cluster (N,2) points with DBSCAN(eps=radius, min_samples=1) and return cluster medians."""
    from sklearn.cluster import DBSCAN
    if len(points) == 0:
        return points
    db = DBSCAN(eps=radius_m, min_samples=1).fit(points)
    labels = db.labels_
    out = []
    for k in np.unique(labels):
        if k == -1:
            continue
        sel = points[labels == k]
        out.append(np.median(sel, axis=0))
    return np.asarray(out, dtype=np.float64)


def build_candidates(bbox_wgs84: Tuple[float, float, float, float],
                     network_type: str = "drive",
                     dedupe_radius_m: float = 80.0,
                     cache_path: str | None = None,
                     target_crs: str = "EPSG:32650") -> StreetCentroids:
    """End-to-end: bbox -> projected & deduped street-centroid candidate pool."""
    G = fetch_street_graph(bbox_wgs84, network_type, cache_path)
    mids = segment_midpoints(G, target_crs)
    deduped = dbscan_dedupe(mids, radius_m=dedupe_radius_m)
    return StreetCentroids(xy_m=deduped, crs=target_crs, n_segments_total=len(mids))


def save_candidates(candidates: StreetCentroids, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        xy_m=candidates.xy_m,
        crs=candidates.crs,
        n_segments_total=np.asarray(candidates.n_segments_total),
    )


def load_candidates(path: str | Path) -> StreetCentroids:
    z = np.load(path, allow_pickle=False)
    return StreetCentroids(
        xy_m=z["xy_m"],
        crs=str(z["crs"]),
        n_segments_total=int(z["n_segments_total"]),
    )
