from math import ceil
from typing import Iterable, Tuple, Optional, Any, Dict, List, Union
import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont

try:
    from ladybug.color import Colorset  # optional
except Exception:
    Colorset = None  # gracefully fallback to a static palette

try:
    from IPython.display import display  # optional, for convenient inline display
except Exception:  # pragma: no cover - IPython not always present
    def display(_):  # type: ignore
        return None


_MODEL_TITLES_OMITTED = object()

# Optional, ergonomic view name presets.
VIEW_PRESETS: Dict[str, Union[str, Tuple[float, float]]] = {
    "iso": "iso",
    "top": (90.0, 0.0),
    "bottom": (-90.0, 0.0),
    "front": (0.0, 0.0),
    "back": (0.0, 180.0),
    "left": (0.0, -90.0),
    "right": (0.0, 90.0),
}


def plot_honeybee_models(
    models: Iterable[Any],
    model_titles: Any = _MODEL_TITLES_OMITTED,
    main_title: Optional[str] = None,
    grid: Optional[Tuple[int, int]] = (6, 2),
    extrude_eps: float = 0.02,
    figsize: Tuple[int, int] = (640, 420),
    total_width: int = 1200,
    base_model_height: int = 600,
    plot_main_legend: bool = True,
    *,
    show=True,
    view: Any = "iso",
    camera_zoom: float = 1.25,
    background_color: Optional[str] = None,
    legend_labels: Optional[List[str]] = None,
    **plot_kwargs,
) -> Optional[Image.Image]:
    """Render multiple Honeybee models into a single image.

    This function renders each model into its own off-screen PyVista plotter and
    then pastes the resulting images onto one canvas in a grid layout. It avoids
    the common artifact of black gutters around subplots by doing the stitching
    manually at the image level.

    Dependencies
    ------------
    - numpy
    - pyvista
    - pillow (PIL)


    Parameters
    ----------
    models :
        Iterable of Honeybee `Model`-like objects to render.
    model_titles :
        Sequence of per-tile titles. If omitted (`_MODEL_TITLES_OMITTED`), use
        each model's `display_name`/`identifier`. If `None`, omit titles entirely.
    main_title : str or None, optional
        If truthy, draw a title strip at the top; if `None`/empty, omit it.
    grid : (int, int) or None, optional
        Rows, cols layout. If `None`, inferred from `len(models)`.
    extrude_eps : float, optional
        Half-thickness (m) used to extrude apertures/doors to avoid z-fighting.
    figsize : (int, int), optional
        Per-tile render size in pixels.
    total_width : int, optional
        Width of the final stitched image in pixels.
    base_model_height : int, optional
        Height of each tile area (excl. title/legend strips) in pixels.
    plot_main_legend : bool, optional
        Whether to draw a global legend for geometry types at the bottom.
    show : bool, optional
        If True (default), display the figure immediately.
        If False, return the PIL.Image for saving or post-processing.
    view : {"iso","top","bottom","front","back","left","right"} or (float,float), optional
        Predefined view name or (elev, azim) tuple.
    camera_zoom : float, optional
        Multiplicative zoom applied after fitting.
    background_color : str or None, optional
        Background color passed to PyVista plotters; `None` uses library default.
    legend_labels : list[str] or None, optional
        Override labels for legend entries.
    **plot_kwargs :
        Forwarded to each `pv.Plotter.add_mesh` call (e.g., `show_wireframe=True`,
        `surface_opacity=0.6`, `wireframe_line_width=1.0`, `wireframe_show_vertices=False`).

    Returns
    -------
    PIL.Image or None
        - If show=False → returns the figure as a PIL.Image
        - If show=True → displays inline and returns None

    Raises
    ------
    TypeError
        If any item in `models` does not expose the expected Honeybee geometry.
    RuntimeError
        If off-screen rendering fails (e.g., missing EGL/OSMesa).


    Examples
    --------

    from honeybee.model import Model
    models = [Model.from_hbjson(p) for p in hbjson_paths]

    # Display inline in notebooks (default behavior)
    plot_honeybee_models(models, main_title="Many models")

    # Or save to disk when not showing interactively
    img = plot_honeybee_models(sample_models, main_title="Many models", show=False)
    img.save("hbmodels.png")
    print(f"Saved hbmodels.png")



    See Also
    --------
    plot_honeybee_model : Interactive Plotly renderer for a single model.

    """

    import numpy as np

    # --------------------- optional Ladybug palette --------------------
    try:
        from ladybug.color import Colorset  # optional
    except Exception:
        Colorset = None  # type: ignore

    def _palette_hex(i: int, fallback: str = "#cccccc") -> str:
        # Prefer Ladybug's OpenStudio palette when available; tolerate API drift
        if Colorset is None:
            base = [
                "#8da0cb", "#fc8d62", "#66c2a5", "#ffd92f",
                "#a6d854", "#e78ac3", "#8dd3c7", "#bebada",
            ]
            return base[i % len(base)]
        try:
            cs = None
            for getter in ("openstudio_palette", "openstudio"):
                if hasattr(Colorset, getter):
                    cs = getattr(Colorset, getter)()
                    break
            if cs is None:
                return fallback
            palette = getattr(cs, "colors", cs)
            col = palette[i % len(palette)]
            return col.to_hex() if hasattr(col, "to_hex") else (getattr(col, "hex", str(col)))
        except Exception:
            return fallback

    COLOR_MAP = {
        "exterior_wall": _palette_hex(0),
        "interior_wall": _palette_hex(1),
        "roof": _palette_hex(3),
        "ceiling": _palette_hex(4),
        "exterior_floor": _palette_hex(6),
        "interior_floor": _palette_hex(7),
        "air_wall": _palette_hex(12),
        "aperture": _palette_hex(9),
        "interior_aperture": _palette_hex(9),
        "door": _palette_hex(10),
        "interior_door": _palette_hex(10),
        "outdoor_shade": _palette_hex(11),
        "indoor_shade": _palette_hex(11),
        "shade_mesh": _palette_hex(11),
        "shade": _palette_hex(11),
        "default": _palette_hex(0),
    }
    STRUCTURE_KEYS = {
        "exterior_floor",
        "interior_floor",
        "roof",
        "ceiling",
        "exterior_wall",
        "interior_wall",
        "air_wall",
    }

    # ----------------------------- title selection -----------------------------

    def _default_titles(_models, _model_titles):
        # model_titles None -> no titles; omitted sentinel -> derive from model names
        if _model_titles is None:
            return [None] * len(_models)
        if _model_titles is _MODEL_TITLES_OMITTED:
            out = []
            for i, m in enumerate(_models):
                t = getattr(m, "display_name", None) or getattr(m, "identifier", None) or f"Model_{i+1}"
                out.append(str(t))
            return out
        if len(_model_titles) != len(_models):
            raise ValueError("model_titles must match number of models.")
        return [None if t is None else str(t) for t in _model_titles]


    # ----------------------------- grid helpers -----------------------------


    def _auto_grid(n_models: int, _figsize: Tuple[int, int], _total_width: int):
        ref_w = max(1, int(_figsize[0]))
        cols = max(1, min(n_models, int(round(_total_width / float(ref_w))) or 1))
        rows = ceil(n_models / cols)
        return cols, rows

    def _per_cell_size(cols: int, _figsize: Tuple[int, int], _total_width: int, _base_model_height: int):
        cw = max(1, int(round(_total_width / float(cols))))
        scale = cw / float(max(1, int(_figsize[0])))
        ch = max(1, int(round(_base_model_height * scale)))
        return cw, ch

    # --- numerics/geometry helpers ---

    def _pt_to_tuple(p):
        if hasattr(p, "x") and hasattr(p, "y") and hasattr(p, "z"):
            return (float(p.x), float(p.y), float(p.z))
        return (float(p[0]), float(p[1]), float(p[2]))

    def _remove_consecutive_duplicates(pts, tol=1e-9):
        if not pts:
            return []
        cleaned = [tuple(pts[0])]
        for p in pts[1:]:
            prev = cleaned[-1]
            if (
                abs(prev[0] - p[0]) <= tol
                and abs(prev[1] - p[1]) <= tol
                and abs(prev[2] - p[2]) <= tol
            ):
                continue
            cleaned.append(tuple(p))
        if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
            cleaned = cleaned[:-1]
        return cleaned

    def _newell_normal(pts3):
        # Robust polygon normal for potentially non-planar inputs
        nx = ny = nz = 0.0
        n = len(pts3)
        for i in range(n):
            x1, y1, z1 = pts3[i]
            x2, y2, z2 = pts3[(i + 1) % n]
            nx += (y1 - y2) * (z1 + z2)
            ny += (z1 - z2) * (x1 + x2)
            nz += (x1 - x2) * (y1 + y2)
        v = np.array((nx, ny, nz), float)
        norm = np.linalg.norm(v)
        return (v / norm) if norm > 1e-12 else np.array([0, 0, 1.0])

    def _build_basis(n):
        # Build a stable 2D basis (u, v) lying in the polygon plane defined by normal n
        n = np.asarray(n, float)
        a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
        u = np.cross(n, a)
        un = np.linalg.norm(u)
        u = np.array([1.0, 0, 0]) if un < 1e-12 else (u / un)
        v = np.cross(n, u)
        vn = np.linalg.norm(v)
        v = np.array([0, 0, 1.0]) if vn < 1e-12 else (v / vn)
        return u, v

    def _project_to_2d(pts3, origin, u, v):
        origin = np.asarray(origin, float)
        return [
            (float(np.dot(np.asarray(p) - origin, u)), float(np.dot(np.asarray(p) - origin, v)))
            for p in pts3
        ]

    def _remove_colinear_2d_with_indices(pts2, indices, rel_tol=1e-12):
        # Drops nearly colinear points to make triangulation robust; retains original indices
        if len(pts2) < 4:
            return pts2[:], indices[:]
        xs = [p[0] for p in pts2]
        ys = [p[1] for p in pts2]
        scale = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
        tol = rel_tol * scale
        cleaned_pts = []
        cleaned_idx = []
        n = len(pts2)
        for i in range(n):
            prev = pts2[(i - 1) % n]
            cur = pts2[i]
            nxt = pts2[(i + 1) % n]
            area = (cur[0] - prev[0]) * (nxt[1] - prev[1]) - (cur[1] - prev[1]) * (nxt[0] - prev[0])
            if abs(area) <= tol:
                continue
            cleaned_pts.append(cur)
            cleaned_idx.append(indices[i])
        if len(cleaned_pts) > 1 and cleaned_pts[0] == cleaned_pts[-1]:
            cleaned_pts = cleaned_pts[:-1]
            cleaned_idx = cleaned_idx[:-1]
        if len(cleaned_pts) < 3:
            # as a last resort, deduplicate to at least 3 vertices
            uniq = []
            uniq_idx = []
            seen = set()
            for p, idx in zip(pts2, indices):
                key = (round(p[0], 12), round(p[1], 12))
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(p)
                uniq_idx.append(idx)
                if len(uniq) >= 3:
                    break
            return (uniq, uniq_idx) if len(uniq) >= 3 else (pts2[:], indices[:])
        return cleaned_pts, cleaned_idx

    def _earclip_with_indices(pts2, indices):
        # Simple ear-clipping triangulation that preserves mapping to original 3D vertices
        def _area2(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        def _inside(p, a, b, c):
            a0 = _area2(p, a, b)
            a1 = _area2(p, b, c)
            a2 = _area2(p, c, a)
            neg = (a0 < -1e-12) or (a1 < -1e-12) or (a2 < -1e-12)
            pos = (a0 > 1e-12) or (a1 > 1e-12) or (a2 > 1e-12)
            return not (neg and pos)

        n = len(pts2)
        if n < 3:
            return []
        if n == 3:
            return [(int(indices[0]), int(indices[1]), int(indices[2]))]
        pts2 = list(pts2)
        indices = list(indices)
        signed_area = sum(pts2[i][0] * pts2[(i + 1) % n][1] - pts2[(i + 1) % n][0] * pts2[i][1] for i in range(n))
        if signed_area < 0:
            pts2.reverse()
            indices.reverse()
        V = list(range(len(pts2)))
        tris = []
        guard = 0
        cap = len(V) * len(V)
        while len(V) > 3 and guard < cap:
            guard += 1
            ear = False
            m = len(V)
            for vi in range(m):
                i0 = V[(vi - 1) % m]
                i1 = V[vi]
                i2 = V[(vi + 1) % m]
                a = pts2[i0]
                b = pts2[i1]
                c = pts2[i2]
                if abs(_area2(a, b, c)) <= 1e-12:
                    continue
                if _area2(a, b, c) <= 0:
                    continue
                if any(_inside(pts2[j], a, b, c) for j in V if j not in (i0, i1, i2)):
                    continue
                tris.append((int(indices[i0]), int(indices[i1]), int(indices[i2])))
                V.pop(vi)
                ear = True
                break
            if not ear:
                for t in range(1, len(V) - 1):
                    tris.append((int(indices[V[0]]), int(indices[V[t]]), int(indices[V[t + 1]])))
                V = []
                break
        if len(V) == 3:
            tris.append((int(indices[V[0]]), int(indices[V[1]]), int(indices[V[2]])))
        return tris

    # --- geometry collectors ---

    def _collect_hb_triangles(model, *, extrude_eps: float, include_types=None, room_ids=None, want_wire=False):
        """Collect triangles per category and wireframe edges for a Honeybee model.

        Returns `(cats, edges_struct, edges_openings)` where `cats` maps category name
        to a dict with packed `pts` and triangle indices `I, J, K` suitable for
        constructing a `pv.PolyData` with faces.
        """
        cats = {}

        def _ensure(cat):
            if cat not in cats:
                cats[cat] = {"pts": [], "idx": {}, "I": [], "J": [], "K": []}
            return cats[cat]

        # separate edges for styling
        edges_struct = []
        edges_openings = []

        OPENING_KEYS = {
            "aperture",
            "interior_aperture",
            "door",
            "interior_door",
            "outdoor_shade",
            "indoor_shade",
            "shade",
            "shade_mesh",
        }

        def _add_wire_polygon(pts3, key=None):
            if not want_wire or not pts3:
                return
            for i in range(len(pts3)):
                a = _pt_to_tuple(pts3[i])
                b = _pt_to_tuple(pts3[(i + 1) % len(pts3)])
                if key in OPENING_KEYS:
                    edges_openings.append((a, b))
                else:
                    edges_struct.append((a, b))

        def _add_polygon(pts3, key):
            if include_types is not None and key not in include_types:
                return
            pts3 = _remove_consecutive_duplicates(pts3)
            if len(pts3) < 3:
                return
            cat = _ensure(key)
            local = []
            for p in pts3:
                k = (round(p[0], 6), round(p[1], 6), round(p[2], 6))
                if k in cat["idx"]:
                    local.append(cat["idx"][k])
                else:
                    gi = len(cat["pts"])
                    cat["pts"].append(k)
                    cat["idx"][k] = gi
                    local.append(gi)
            verts = [cat["pts"][i] for i in local]
            try:
                n = _newell_normal(verts)
                u, v = _build_basis(n)
                org = np.array(verts[0], float)
                pts2 = _project_to_2d(verts, org, u, v)
                pts2, idxs = _remove_colinear_2d_with_indices(pts2, local)
                if len(pts2) < 3:
                    for t in range(1, len(local) - 1):
                        cat["I"] += [local[0]]
                        cat["J"] += [local[t]]
                        cat["K"] += [local[t + 1]]
                else:
                    for (a, b, c) in _earclip_with_indices(pts2, idxs):
                        cat["I"] += [a]
                        cat["J"] += [b]
                        cat["K"] += [c]
            finally:
                if want_wire:
                    _add_wire_polygon(pts3, key)

        def _add_mesh_like(mesh_geom, key):
            # Accepts geometry objects that expose vertices and faces (triangles or polys)
            if include_types is not None and key not in include_types:
                return
            if mesh_geom is None:
                return
            verts = getattr(mesh_geom, "vertices", None) or getattr(mesh_geom, "points", None) or getattr(mesh_geom, "vertices_xyz", None)
            faces = getattr(mesh_geom, "faces", None) or getattr(mesh_geom, "triangles", None) or getattr(mesh_geom, "indices", None)
            if not verts or not faces:
                return
            v3 = [_pt_to_tuple(p) for p in verts]
            cat = _ensure(key)
            for p in v3:
                k = (round(p[0], 6), round(p[1], 6), round(p[2], 6))
                if k not in cat["idx"]:
                    gi = len(cat["pts"])
                    cat["pts"].append(k)
                    cat["idx"][k] = gi
            for f in faces:
                try:
                    idxs = list(f)
                except Exception:
                    continue
                if len(idxs) < 3:
                    continue
                a = idxs[0]
                for t in range(1, len(idxs) - 1):
                    b = idxs[t]
                    c = idxs[t + 1]
                    pa, pb, pc = v3[a], v3[b], v3[c]
                    ia = cat["idx"][(round(pa[0], 6), round(pa[1], 6), round(pa[2], 6))]
                    ib = cat["idx"][(round(pb[0], 6), round(pb[1], 6), round(pb[2], 6))]
                    ic = cat["idx"][(round(pc[0], 6), round(pc[1], 6), round(pc[2], 6))]
                    cat["I"].append(ia)
                    cat["J"].append(ib)
                    cat["K"].append(ic)

        def _add_extruded(pts3, key, eps):
            if include_types is not None and key not in include_types:
                return
            pts3 = _remove_consecutive_duplicates(pts3)
            if len(pts3) < 3:
                return
            n = _newell_normal(pts3)
            cat = _ensure(key)
            front = []
            back = []
            for p in pts3:
                vf = (
                    round(p[0] + n[0] * eps, 6),
                    round(p[1] + n[1] * eps, 6),
                    round(p[2] + n[2] * eps, 6),
                )
                vb = (
                    round(p[0] - n[0] * eps, 6),
                    round(p[1] - n[1] * eps, 6),
                    round(p[2] - n[2] * eps, 6),
                )
                for vtx, acc in ((vf, front), (vb, back)):
                    if vtx in cat["idx"]:
                        acc.append(cat["idx"][vtx])
                    else:
                        gi = len(cat["pts"])
                        cat["pts"].append(vtx)
                        cat["idx"][vtx] = gi
                        acc.append(gi)
            n_ = len(front)
            if n_ < 3:
                return
            # front face
            verts = [cat["pts"][i] for i in front]
            u, v = _build_basis(_newell_normal(verts))
            org = np.array(verts[0], float)
            pts2 = _project_to_2d(verts, org, u, v)
            pts2, idxs = _remove_colinear_2d_with_indices(pts2, front)
            if len(pts2) < 3:
                for t in range(1, n_ - 1):
                    cat["I"] += [front[0]]
                    cat["J"] += [front[t]]
                    cat["K"] += [front[t + 1]]
            else:
                for (a, b, c) in _earclip_with_indices(pts2, idxs):
                    cat["I"] += [a]
                    cat["J"] += [b]
                    cat["K"] += [c]
            # back face (reversed)
            for t in range(1, n_ - 1):
                cat["I"] += [back[0]]
                cat["J"] += [back[t + 1]]
                cat["K"] += [back[t]]
            # side quads split into two triangles
            for i in range(n_):
                ni = (i + 1) % n_
                a = front[i]
                b = front[ni]
                c = back[ni]
                d = back[i]
                cat["I"] += [a, a]
                cat["J"] += [b, c]
                cat["K"] += [c, d]
            if want_wire:
                _add_wire_polygon(pts3, key)

        def _bc_name(bc):
            if bc is None:
                return ""
            for attr in ("name", "type", "display_name"):
                v = getattr(bc, attr, None)
                if v:
                    return str(v).lower()
            if getattr(bc, "boundary_condition_objects", None):
                return "surface"
            return ""

        def _surface_category(face):
            tname = (
                str(getattr(face.type, "name", "")).lower() if getattr(face, "type", None) else ""
            )
            bc = _bc_name(getattr(face, "boundary_condition", None))
            is_surface = "surface" in bc
            is_adiabatic = "adiabatic" in bc
            is_outdoor = ("outdoor" in bc or bc == "outdoors")
            is_ground = "ground" in bc
            if "airboundary" in tname:
                return "air_wall"
            if ("roof" in tname or "roofceiling" in tname):
                return "ceiling" if (is_surface or is_adiabatic) else "roof"
            if "floor" in tname:
                return "exterior_floor" if (is_outdoor or is_ground) else "interior_floor"
            if "wall" in tname:
                if is_outdoor:
                    return "exterior_wall"
                if is_surface or is_adiabatic:
                    return "interior_wall"
                return "interior_wall"
            return "default"

        def _ap_cat(ap, host):
            bc = _bc_name(getattr(ap, "boundary_condition", None))
            if "outdoor" in bc or bc == "outdoors":
                return "aperture"
            if "surface" in bc or getattr(
                getattr(ap, "boundary_condition", None), "boundary_condition_objects", None
            ):
                return "interior_aperture"
            return "interior_aperture" if host == "interior_wall" else "aperture"

        def _door_cat(dr, host):
            bc = _bc_name(getattr(dr, "boundary_condition", None))
            if "outdoor" in bc or bc == "outdoors":
                return "door"
            if "surface" in bc or getattr(
                getattr(dr, "boundary_condition", None), "boundary_condition_objects", None
            ):
                return "interior_door"
            return "interior_door" if host == "interior_wall" else "door"

        # filter rooms up-front if requested
        all_rooms = getattr(model, "rooms", []) or []
        rooms = (
            [rm for rm in all_rooms if getattr(rm, "identifier", None) in set(room_ids)]
            if room_ids
            else all_rooms
        )

        from itertools import chain

        for rm in rooms:
            for face in getattr(rm, "faces", []) or []:
                geom = getattr(face, "geometry", None)
                raw = getattr(geom, "boundary", None) if geom is not None else None
                if raw is None and geom is not None:
                    raw = getattr(geom, "vertices", None) or getattr(geom, "points", None) or getattr(geom, "coordinates", None)
                if raw:
                    pts = [_pt_to_tuple(p) for p in raw]
                    _add_polygon(pts, _surface_category(face))
                # shades attached to face
                for sh in chain(getattr(face, "outdoor_shades", []) or (), getattr(face, "indoor_shades", []) or ()):  # noqa: E501
                    sg = getattr(sh, "geometry", None)
                    if not sg:
                        continue
                    raw_s = getattr(sg, "boundary", None) or getattr(sg, "vertices", None) or getattr(sg, "points", None)
                    key = "indoor_shade" if getattr(sh, "is_indoor", False) else "outdoor_shade"
                    if raw_s:
                        _add_polygon([_pt_to_tuple(p) for p in raw_s], key)
                    else:
                        _add_mesh_like(sg, "shade_mesh")
                # apertures
                for ap in getattr(face, "apertures", []) or []:
                    # shades attached to aperture
                    for sh in chain(getattr(ap, "outdoor_shades", []) or (), getattr(ap, "indoor_shades", []) or ()):  # noqa: E501
                        sg = getattr(sh, "geometry", None)
                        if not sg:
                            continue
                        raw_s = getattr(sg, "boundary", None) or getattr(sg, "vertices", None) or getattr(sg, "points", None)
                        skey = "indoor_shade" if getattr(sh, "is_indoor", False) else "outdoor_shade"
                        if raw_s:
                            _add_polygon([_pt_to_tuple(p) for p in raw_s], skey)
                        else:
                            _add_mesh_like(sg, "shade_mesh")
                    ag = getattr(ap, "geometry", None)
                    raw_a = (
                        getattr(ag, "boundary", None) or getattr(ag, "vertices", None) or getattr(ag, "points", None)
                        if ag
                        else None
                    )
                    host = _surface_category(face)
                    key = _ap_cat(ap, host)
                    if raw_a:
                        _add_extruded([_pt_to_tuple(p) for p in raw_a], key, extrude_eps)
                    else:
                        _add_mesh_like(ag, key)
                # doors
                for dr in getattr(face, "doors", []) or []:
                    # shades attached to door
                    for sh in chain(getattr(dr, "outdoor_shades", []) or (), getattr(dr, "indoor_shades", []) or ()):  # noqa: E501
                        sg = getattr(sh, "geometry", None)
                        if not sg:
                            continue
                        raw_s = getattr(sg, "boundary", None) or getattr(sg, "vertices", None) or getattr(sg, "points", None)
                        skey = "indoor_shade" if getattr(sh, "is_indoor", False) else "outdoor_shade"
                        if raw_s:
                            _add_polygon([_pt_to_tuple(p) for p in raw_s], skey)
                        else:
                            _add_mesh_like(sg, "shade_mesh")
                    dg = getattr(dr, "geometry", None)
                    raw_d = (
                        getattr(dg, "boundary", None) or getattr(dg, "vertices", None) or getattr(dg, "points", None)
                        if dg
                        else None
                    )
                    host = _surface_category(face)
                    key = _door_cat(dr, host)
                    if raw_d:
                        _add_extruded([_pt_to_tuple(p) for p in raw_d], key, extrude_eps)
                    else:
                        _add_mesh_like(dg, key)

        # orphaned shades and shade meshes
        for sh in getattr(model, "orphaned_shades", []) or []:
            sg = getattr(sh, "geometry", None)
            if sg:
                raw = getattr(sg, "boundary", None) or getattr(sg, "vertices", None) or getattr(sg, "points", None)
                cat = "indoor_shade" if getattr(sh, "is_indoor", False) else "outdoor_shade"
                if raw:
                    _add_polygon([_pt_to_tuple(p) for p in raw], cat)
                else:
                    _add_mesh_like(sg, "shade_mesh")
        for sm in getattr(model, "shade_meshes", []) or []:
            smg = getattr(sm, "geometry", None) or getattr(sm, "mesh", None) or sm
            _add_mesh_like(smg, "shade_mesh")

        # orphaned faces (+ hosted apertures/doors)
        for face in getattr(model, "orphaned_faces", []) or []:
            geom = getattr(face, "geometry", None)
            raw = (
                getattr(geom, "boundary", None) or getattr(geom, "vertices", None) or getattr(geom, "points", None)
                if geom
                else None
            )
            if raw:
                fcat = _surface_category(face)
                _add_polygon([_pt_to_tuple(p) for p in raw], fcat)
                # hosted apertures on orphaned face
                for ap in getattr(face, "apertures", []) or []:
                    ag = getattr(ap, "geometry", None)
                    raw_a = getattr(ag, "boundary", None) or getattr(ag, "vertices", None) or getattr(ag, "points", None) if ag else None
                    k = _ap_cat(ap, fcat)
                    if raw_a:
                        _add_extruded([_pt_to_tuple(p) for p in raw_a], k, extrude_eps)
                    else:
                        _add_mesh_like(ag, k)
                # hosted doors on orphaned face
                for dr in getattr(face, "doors", []) or []:
                    dg = getattr(dr, "geometry", None)
                    raw_d = getattr(dg, "boundary", None) or getattr(dg, "vertices", None) or getattr(dg, "points", None) if dg else None
                    k = _door_cat(dr, fcat)
                    if raw_d:
                        _add_extruded([_pt_to_tuple(p) for p in raw_d], k, extrude_eps)
                    else:
                        _add_mesh_like(dg, k)

        # orphaned apertures and doors
        for ap in getattr(model, "orphaned_apertures", []) or []:
            ag = getattr(ap, "geometry", None)
            if ag:
                raw = getattr(ag, "boundary", None) or getattr(ag, "vertices", None) or getattr(ag, "points", None)
                k = _ap_cat(ap, "exterior_wall")
                if raw:
                    _add_extruded([_pt_to_tuple(p) for p in raw], k, extrude_eps)
                else:
                    _add_mesh_like(ag, k)
        for dr in getattr(model, "orphaned_doors", []) or []:
            dg = getattr(dr, "geometry", None)
            if dg:
                raw = getattr(dg, "boundary", None) or getattr(dg, "vertices", None) or getattr(dg, "points", None)
                k = _door_cat(dr, "exterior_wall")
                if raw:
                    _add_extruded([_pt_to_tuple(p) for p in raw], k, extrude_eps)
                else:
                    _add_mesh_like(dg, k)

        return cats, edges_struct, edges_openings

    def _polydata_from_category(cat):
        # Convert category buckets into a faces-based PolyData mesh
        pts = np.asarray(cat["pts"], dtype=np.float32)
        if pts.size == 0 or len(cat["I"]) == 0:
            return None
        I = np.asarray(cat["I"], dtype=np.int64)
        J = np.asarray(cat["J"], dtype=np.int64)
        K = np.asarray(cat["K"], dtype=np.int64)
        faces = np.column_stack([np.full(I.shape[0], 3, dtype=np.int64), I, J, K]).ravel()
        return pv.PolyData(pts, faces=faces)

    def _wire_polydata(edges):
        # Build a PolyData that contains only line cells; ensure there are no
        # vertex or face cells so PyVista won't render points at vertices.
        if not edges:
            return None
        idx = {}
        pts = []
        lines = []

        def addp(p):
            k = (round(p[0], 6), round(p[1], 6), round(p[2], 6))
            if k in idx:
                return idx[k]
            i = len(pts)
            idx[k] = i
            pts.append([p[0], p[1], p[2]])
            return i

        for a, b in edges:
            ia = addp(a)
            ib = addp(b)
            lines.extend([2, ia, ib])
        pts_arr = np.asarray(pts, np.float32)
        lines_arr = np.asarray(lines, np.int64)
        pd = pv.PolyData(pts_arr, lines=lines_arr)
        try:
            pd.verts = np.empty(0, dtype=np.int64)
            pd.faces = np.empty(0, dtype=np.int64)
        except Exception:
            pass
        return pd

    # --------------------------- inputs and layout -----------------------------

    models = list(models)
    if not models:
        raise ValueError("No models provided.")

    # forwarded options from plot_kwargs
    show_wireframe = bool(plot_kwargs.get("show_wireframe", False))
    surface_opacity = float(plot_kwargs.get("surface_opacity", 1))
    include_types = set(plot_kwargs["include_types"]) if plot_kwargs.get("include_types") is not None else None
    room_ids = plot_kwargs.get("room_ids", None)
    wire_color = plot_kwargs.get("wireframe_color", "black")
    wire_width = float(plot_kwargs.get("wireframe_width", 1.5))
    # openings (windows/doors/shades) wireframe style — default slightly thinner
    wire_open_color = plot_kwargs.get("wireframe_openings_color", wire_color)
    wire_open_width = float(plot_kwargs.get("wireframe_openings_width", max(0.5, wire_width * 0.7)))
    # show or hide vertex points at wireframe corners (default: hide)
    wire_show_vertices = bool(plot_kwargs.get("wireframe_show_vertices", False))
    # prefer passed background_color argument but fall back to plot_kwargs then white
    bg_color = background_color or plot_kwargs.get("background_color", "white")
    aspectmode = plot_kwargs.get("aspectmode", "data")

    # grid layout
    if grid is None:
        cols, rows = _auto_grid(len(models), figsize, total_width)
    else:
        cols, rows = int(grid[0]), int(grid[1])
        max_cells = max(1, cols * rows)
        if max_cells < len(models):
            models = models[:max_cells]
    cell_w, cell_h = _per_cell_size(cols, figsize, total_width, base_model_height)
    total_h = rows * cell_h

    titles = _default_titles(models, model_titles)

    # --- Title and legend measurements (with robust font fallback) ---

    def _load_truetype_font(size: int):
        # Try several common fonts across platforms so the title can be drawn large
        candidates = [
            "DejaVuSans.ttf",
            "Arial.ttf",
            "LiberationSans-Regular.ttf",
            "FreeSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            "/Library/Fonts/Arial.ttf",
            "C:/Windows/Fonts/Arial.ttf",
        ]
        for name in candidates:
            try:
                return ImageFont.truetype(name, size)
            except Exception:
                continue
        return None

    # choose reasonable title font size proportional to canvas
    title_font_size = max(18, int(min(total_width, total_h) / 16))
    title_font = _load_truetype_font(title_font_size)
    if title_font is None:
        # if no TTF available, fallback to default; approximate a larger logical size
        title_font = ImageFont.load_default()
        title_font_size = getattr(title_font, "size", 12) * max(6, title_font_size // 12)

    # compute top title strip height; 0 if no main_title
    title_h = 0
    if main_title:
        test_img = Image.new("RGBA", (10, 10))
        test_draw = ImageDraw.Draw(test_img)
        try:
            bbox = test_draw.textbbox((0, 0), str(main_title), font=title_font)
            title_h = bbox[3] - bbox[1] + 32  # base padding for title
        except Exception:
            try:
                _, h = test_draw.textsize(str(main_title), font=title_font)
                title_h = h + 32
            except Exception:
                title_h = title_font_size + 32
        title_grid_margin = max(20, title_font_size // 2)  # breathing room below title
        title_h += title_grid_margin

    # Legend measurement and layout parameters
    legend_h = 0
    legend_padding = 10
    legend_item_gap = 12
    legend_square = 12
    try:
        legend_font_size = max(12, int(title_font_size * 0.55))
        legend_font = ImageFont.truetype("DejaVuSans.ttf", legend_font_size)
    except Exception:
        legend_font = ImageFont.load_default()
        legend_font_size = getattr(legend_font, "size", 12)

    default_order = [
        "exterior_floor",
        "interior_floor",
        "roof",
        "ceiling",
        "exterior_wall",
        "interior_wall",
        "air_wall",
        "aperture",
        "interior_aperture",
        "door",
        "interior_door",
        "outdoor_shade",
        "indoor_shade",
        "shade_mesh",
        "shade",
        "default",
    ]
    keys_in_map = list(COLOR_MAP.keys())
    ordered = []
    for k in default_order:
        if k in keys_in_map and k not in ordered:
            ordered.append(k)
    for k in keys_in_map:
        if k not in ordered:
            ordered.append(k)
    legend_names = legend_labels if legend_labels is not None else ordered

    if plot_main_legend:
        n_items = len(legend_names)
        target_col_w = 220  # target width per legend column
        # Guard against tiny widths and keep within [1..min(4, n_items)]
        safe_width = max(1, int(total_width))
        num_cols = max(1, min(4, n_items, safe_width // target_col_w if target_col_w else 1))
        per_col = int(np.ceil(n_items / float(num_cols))) if num_cols else n_items
        test_img = Image.new("RGBA", (10, 10))
        test_draw = ImageDraw.Draw(test_img)
        max_txt_w = 0
        max_txt_h = 0
        for name in legend_names:
            txt = str(name)
            try:
                bbox = test_draw.textbbox((0, 0), txt, font=legend_font)
                txt_w = bbox[2] - bbox[0]
                txt_h = bbox[3] - bbox[1]
            except Exception:
                try:
                    txt_w, txt_h = test_draw.textsize(txt, font=legend_font)
                except Exception:
                    txt_w = int(len(txt) * legend_font_size * 0.6)
                    txt_h = legend_font_size
            max_txt_w = max(max_txt_w, txt_w)
            max_txt_h = max(max_txt_h, txt_h)
        item_h = max(legend_square, max_txt_h) + 6
        legend_h = per_col * item_h + 2 * legend_padding
    else:
        num_cols = 0
        per_col = 0
        item_h = 0

    # ----------------------- compose the full canvas ---------------------------

    canvas_w = int(total_width)
    canvas_h = int(title_h + total_h + legend_h)
    try:
        big_img = Image.new("RGBA", (canvas_w, canvas_h), color=bg_color)
    except Exception:
        big_img = Image.new("RGBA", (canvas_w, canvas_h), color=(255, 255, 255, 255))

    # Render each model in its own plotter and paste into the grid
    for i, m in enumerate(models):
        r, c = divmod(i, cols)
        p = pv.Plotter(off_screen=True, window_size=(int(cell_w), int(cell_h)))
        p.set_background(bg_color)

        cats, edges_struct, edges_openings = _collect_hb_triangles(
            m,
            extrude_eps=extrude_eps,
            include_types=include_types,
            room_ids=room_ids,
            want_wire=show_wireframe,
        )

        # filled surfaces per category
        for key in sorted(cats.keys()):
            cat = cats[key]
            if not cat["pts"] or not cat["I"]:
                continue
            pd = _polydata_from_category(cat)
            if pd is None:
                continue
            p.add_mesh(
                pd,
                color=COLOR_MAP.get(key, COLOR_MAP["default"]),
                opacity=np.clip(surface_opacity, 0.0, 1.0),
                smooth_shading=False,
                lighting=False,
                show_edges=False,
                edge_color="black",
                line_width=1.0,
                reset_camera=False,
                # Ensure no point glyphs ever appear on filled meshes
                show_vertices=False,
                render_points_as_spheres=False,
                point_size=0,
            )

        # wireframe overlays with separate styling for structural vs openings
        if show_wireframe:
            if edges_struct:
                wpd_s = _wire_polydata(edges_struct)
                if wpd_s is not None:
                    p.add_mesh(
                        wpd_s,
                        color=wire_color,
                        line_width=wire_width,
                        render_lines_as_tubes=False,
                        show_vertices=wire_show_vertices,
                        render_points_as_spheres=False,
                        point_size=0,
                        lighting=False,
                    )
            if edges_openings:
                wpd_o = _wire_polydata(edges_openings)
                if wpd_o is not None:
                    p.add_mesh(
                        wpd_o,
                        color=wire_open_color,
                        line_width=wire_open_width,
                        render_lines_as_tubes=False,
                        show_vertices=wire_show_vertices,
                        render_points_as_spheres=False,
                        point_size=0,
                        lighting=False,
                    )

        # camera handling (use methods, not attribute assignment)
        if isinstance(view, str):
            preset = VIEW_PRESETS.get(view.lower(), "iso")
            if preset == "iso":
                p.view_isometric()
            else:
                elev, azim = preset  # type: ignore[assignment]
                try:
                    p.camera.elevation(float(elev))
                    p.camera.azimuth(float(azim))
                except Exception:
                    # Fall back to isometric if camera API differs
                    p.view_isometric()
        elif isinstance(view, (tuple, list)) and len(view) == 2:
            elev, azim = view
            try:
                p.camera.elevation(float(elev))
                p.camera.azimuth(float(azim))
            except Exception:
                p.view_isometric()
        else:
            p.view_isometric()

        # zoom: in VTK, zoom(z) > 1 zooms in; we keep compatibility with existing API
        p.camera.zoom(1.0 / float(camera_zoom))

        # normalize aspect if requested
        if aspectmode == "cube":
            b = p.bounds
            if b:
                sx, sy, sz = b[1] - b[0], b[3] - b[2], b[5] - b[4]
                s = max(sx, sy, sz)
                cx, cy, cz = (b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2
                cube = pv.Cube(center=(cx, cy, cz), x_length=s, y_length=s, z_length=s)
                p.add_mesh(cube, opacity=0.0, reset_camera=False)

        # per-cell title (small caption in the upper-left)
        if titles[i]:
            p.add_text(
                str(titles[i]),
                position="upper_left",
                font_size=max(10, int(0.8 * min(cell_w, cell_h) / 40)),
            )

        # grab the off-screen buffer and paste
        cell_arr = p.screenshot(return_img=True)
        p.close()
        cell_img = Image.fromarray(cell_arr).convert("RGBA")
        x0 = int(c * cell_w)
        y0 = int(title_h + r * cell_h)
        big_img.paste(cell_img, (x0, y0), cell_img)

    # Draw main title strip (if provided)
    if main_title:
        draw = ImageDraw.Draw(big_img)
        font = title_font
        txt = str(main_title)
        try:
            bbox = draw.textbbox((0, 0), txt, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            try:
                w, h = draw.textsize(txt, font=font)
            except Exception:
                w = int(len(txt) * (getattr(font, "size", 12) * 0.6))
                h = getattr(font, "size", 12)
        x = max(20, (canvas_w - w) // 2)
        y = max(12, (title_h - h) // 2)
        draw.text((x, y), txt, fill="black", font=font)

    # Draw bottom legend (if requested)
    if plot_main_legend:
        draw = ImageDraw.Draw(big_img)
        names = list(legend_names)
        start_y = int(title_h + total_h + legend_padding)
        col_w = int((canvas_w - 2 * legend_padding) / float(num_cols)) if num_cols > 0 else canvas_w - 2 * legend_padding
        for col in range(max(1, num_cols)):
            col_x = legend_padding + col * (col_w if num_cols > 0 else (canvas_w - 2 * legend_padding))
            for row_i in range(per_col if num_cols > 0 else len(names)):
                idx = col * (per_col if num_cols > 0 else len(names)) + row_i
                if idx >= len(names):
                    break
                name = names[idx]
                label = str(name)
                # measure text for vertical centering
                try:
                    bbox = draw.textbbox((0, 0), label, font=legend_font)
                    txt_w = bbox[2] - bbox[0]
                    txt_h = bbox[3] - bbox[1]
                except Exception:
                    try:
                        txt_w, txt_h = draw.textsize(label, font=legend_font)
                    except Exception:
                        txt_w = int(len(label) * legend_font_size * 0.6)
                        txt_h = legend_font_size
                item_y = start_y + row_i * item_h if num_cols > 0 else start_y
                if num_cols == 0:
                    # single-column fallback compaction
                    item_y = start_y + idx * (max(legend_square, txt_h) + 6)
                sq_x0 = col_x
                sq_y0 = item_y + (max(legend_square, txt_h) - legend_square) // 2
                sq_x1 = sq_x0 + legend_square
                sq_y1 = sq_y0 + legend_square
                color = COLOR_MAP.get(name, COLOR_MAP.get("default", "#cccccc"))
                try:
                    draw.rectangle([sq_x0, sq_y0, sq_x1, sq_y1], fill=color, outline="black")
                except Exception:
                    draw.rectangle([sq_x0, sq_y0, sq_x1, sq_y1], fill=(200, 200, 200), outline="black")
                tx = sq_x1 + 6
                ty = sq_y0 + max(0, (legend_square - txt_h) // 2)
                draw.text((tx, ty), label, fill="black", font=legend_font)

    final_arr = np.asarray(big_img)

    # final stitched figure
    fig: Image.Image = big_img  # the stitched PIL image

    if show:
        # Display once, and return None to avoid Jupyter auto-rendering the return value
        try:
            from IPython.display import display as _display
            _display(fig)
        except Exception:
            try:
                fig.show()
            except Exception:
                pass
        return None

    # When show=False, don't display—return the image for saving/post-processing
    return fig