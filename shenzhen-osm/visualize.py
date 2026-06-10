import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import shape

DIR = Path.home() / "shenzhen-osm"


def load(name):
    g = gpd.read_file(DIR / f"{name}.geojson")
    if "tags" in g.columns:
        # promote common tag keys to top-level columns
        for key in ("highway", "building", "natural", "waterway", "landuse",
                    "amenity", "shop", "tourism", "leisure", "name"):
            g[key] = g["tags"].apply(
                lambda t, k=key: t.get(k) if isinstance(t, dict) else None
            )
    return g


def static_map():
    print("loading layers...", flush=True)
    boundary = load("boundary")
    water = load("water")
    landuse = load("landuse")
    roads = load("roads")
    buildings = load("buildings")

    fig, ax = plt.subplots(figsize=(16, 16), dpi=150)
    ax.set_facecolor("#f4f1ea")

    # landuse — light, beneath everything
    if not landuse.empty:
        landuse.plot(ax=ax, color="#e6e2d3", edgecolor="none", alpha=0.6)

    # water
    if not water.empty:
        water.plot(ax=ax, color="#a6cee3", edgecolor="#7fb3d5", linewidth=0.3)

    # buildings — dark fill, no edge for speed
    if not buildings.empty:
        buildings.plot(ax=ax, color="#8c8a85", edgecolor="none", alpha=0.7)

    # roads — split by class
    if not roads.empty:
        major = roads[roads["highway"].isin(
            ["motorway", "trunk", "primary", "motorway_link", "trunk_link"]
        )]
        secondary = roads[roads["highway"].isin(["secondary", "tertiary"])]
        minor = roads[~roads["highway"].isin(
            ["motorway", "trunk", "primary", "motorway_link", "trunk_link",
             "secondary", "tertiary"]
        )]
        if not minor.empty:
            minor.plot(ax=ax, color="#b8b8b8", linewidth=0.2)
        if not secondary.empty:
            secondary.plot(ax=ax, color="#666666", linewidth=0.5)
        if not major.empty:
            major.plot(ax=ax, color="#d62728", linewidth=0.9)

    # boundary — outline on top
    boundary.boundary.plot(ax=ax, color="black", linewidth=1.2)

    minx, miny, maxx, maxy = boundary.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Shenzhen — OpenStreetMap (boundary, water, landuse, roads, buildings)",
                 fontsize=14, pad=12)

    out = DIR / "overview.png"
    fig.savefig(out, bbox_inches="tight", facecolor="#f4f1ea")
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")


def interactive_map():
    import folium

    print("building interactive map...", flush=True)
    boundary = load("boundary")
    pois = load("pois")
    roads = load("roads")

    centroid = boundary.geometry.iloc[0].centroid
    m = folium.Map(
        location=[centroid.y, centroid.x],
        zoom_start=10,
        tiles="CartoDB positron",
    )

    # boundary
    folium.GeoJson(
        boundary[["geometry"]].to_json(),
        name="边界",
        style_function=lambda f: {"color": "black", "weight": 2, "fill": False},
    ).add_to(m)

    # major roads only
    major = roads[roads["highway"].isin(
        ["motorway", "trunk", "primary", "motorway_link", "trunk_link"]
    )]
    folium.GeoJson(
        major[["geometry", "name", "highway"]].to_json(),
        name=f"主干道 ({len(major)})",
        style_function=lambda f: {"color": "#d62728", "weight": 1.5},
    ).add_to(m)

    # sample POIs (cap at ~2000)
    poi_pts = pois[pois.geometry.geom_type == "Point"]
    if len(poi_pts) > 2000:
        poi_pts = poi_pts.sample(2000, random_state=0)

    from folium.plugins import MarkerCluster
    poi_cluster = MarkerCluster(name=f"POI 采样 ({len(poi_pts)}/{len(pois)})")
    for _, row in poi_pts.iterrows():
        name = row.get("name") or row.get("amenity") or row.get("shop") or "?"
        cat = row.get("amenity") or row.get("shop") or row.get("tourism") or row.get("leisure") or ""
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=3,
            popup=f"{name} ({cat})",
            color="#1f77b4",
            fill=True,
            fillOpacity=0.7,
            weight=0.5,
        ).add_to(poi_cluster)
    poi_cluster.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    out = DIR / "interactive.html"
    m.save(str(out))
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    static_map()
    interactive_map()
