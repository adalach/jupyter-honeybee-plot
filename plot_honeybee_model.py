# sentinel used to detect when the caller omitted the `title` argument
_TITLE_OMITTED = object()

def plot_honeybee_model(
    model,
    title=_TITLE_OMITTED,
    *,
    extrude_eps: float = 0.02,
    figsize: tuple[int, int] = (640, 420),
    show: bool = True,
    show_wireframe: bool = False,
    surface_opacity: float = 0.5,
    show_legend: bool = False,
    include_types: list[str] | None = None,
    room_ids: list[str] | None = None,
    background_color: str | None = "white",
    aspectmode: str = "data",
):
    """
    Render a single Honeybee model as an interactive Plotly 3D figure.

    This function visualizes a Honeybee `Model` using Plotly. It colors filled
    surfaces by geometry category, optionally overlays a thin wireframe,
    and slightly extrudes apertures and doors to avoid z-fighting.
    Titles and legends can be toggled or customized. Intended for
    interactive inspection within Jupyter notebooks.


    Parameters
    ----------
    model : honeybee.model.Model
        Honeybee model instance (e.g., `Model.from_hbjson(path)`).
    title : str, None, or _TITLE_OMITTED, optional
        Controls the top-of-plot title.
        - If `title` is omitted (the default sentinel `_TITLE_OMITTED`), the plot title
          is inferred from `model.display_name` or `model.identifier`.
        - If `title` is a string, that string is used.
        - If `title` is `None`, no title is shown and no top margin is reserved.
    extrude_eps : float, optional
        Small half-thickness (m) applied when extruding apertures and doors to
        prevent z-fighting. Default is `0.02`.
    figsize : (int, int), optional
        Figure size in pixels as `(width, height)`. Default is `(640, 420)`.
    show : bool, optional
        If `True`, automatically display the Plotly figure via `fig.show()`.
    show_wireframe : bool, optional
        If `True`, overlay thin line segments for structural surfaces.
    surface_opacity : float, optional
        Opacity of filled surfaces in `[0, 1]`. Default is `0.5`.
    show_legend : bool, optional
        If `True`, display the Plotly legend with geometry categories.
    include_types : list of str or None, optional
        Restrict rendering to a subset of geometry types (e.g.,
        `["Wall", "Roof", "Floor"]`). If `None`, render all types present.
    room_ids : list of str or None, optional
        List of specific Honeybee Room identifiers to include. If `None`, include
        all rooms.
    background_color : str or None, optional
        Plot background color; defaults to `"white"`.
    aspectmode : {"data", "cube", "auto", "manual"}, optional
        Plotly 3D scene aspect ratio mode. `"data"` preserves geometric scale.

    Returns
    -------
    plotly.graph_objects.Figure
        The constructed Plotly figure. Returned even if `show=True`.

    Raises
    ------
    TypeError
        If `model` is not a valid Honeybee `Model` instance.
    ValueError
        If the figure cannot be built due to invalid geometry or empty selection.


    Examples
    --------
    from honeybee.model import Model
    model = Model.from_hbjson("office_building.hbjson")
    plot_honeybee_model(model)

    from honeybee.model import Model
    model = Model.from_hbjson("office_building.hbjson")
    fig = plot_honeybee_model(model, show_wireframe=True, show_legend=True, show=False)
    fig.write_html("office_building.html")


    See Also
    --------
    plot_honeybee_models : Static PyVista renderer for multiple models.
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
    if title is _TITLE_OMITTED:
        title_val = getattr(model, "display_name", None) or getattr(model, "identifier", None)
    else:
        title_val = title

    # ----------------------------- geometry helpers ----------------------------
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

    # --------------------------- classification helpers ------------------------
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

    def _aperture_category(ap, host):
        bc = _bc_name(getattr(ap, "boundary_condition", None))
        if "outdoor" in bc or bc == "outdoors":
            return "aperture"
        if "surface" in bc or getattr(
            getattr(ap, "boundary_condition", None), "boundary_condition_objects", None
        ):
            return "interior_aperture"
        return "interior_aperture" if host == "interior_wall" else "aperture"

    def _door_category(dr, host):
        bc = _bc_name(getattr(dr, "boundary_condition", None))
        if "outdoor" in bc or bc == "outdoors":
            return "door"
        if "surface" in bc or getattr(
            getattr(dr, "boundary_condition", None), "boundary_condition_objects", None
        ):
            return "interior_door"
        return "interior_door" if host == "interior_wall" else "door"

    # -------------------------- storage & triangulation ------------------------
    categories: dict[str, dict] = {}

    def _ensure(cat):
        if cat not in categories:
            categories[cat] = {"pts": [], "idx": {}, "I": [], "J": [], "K": []}
        return categories[cat]

    # optional wireframe (structural edges only)
    edge_segments: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []

    def _add_wire_edges_from_polygon(pts3):
        if not pts3 or len(pts3) < 2:
            return
        for i in range(len(pts3)):
            p = _pt_to_tuple(pts3[i])
            q = _pt_to_tuple(pts3[(i + 1) % len(pts3)])
            edge_segments.append((p, q))

    def _add_polygon(pts3, key):
        if include_types is not None and key not in include_types:
            return
        if not pts3:
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
        except Exception:
            for t in range(1, len(local) - 1):
                cat["I"] += [local[0]]
                cat["J"] += [local[t]]
                cat["K"] += [local[t + 1]]
        if show_wireframe and key in STRUCTURE_KEYS:
            _add_wire_edges_from_polygon(pts3)

    def _add_extruded_polygon(pts3, key, eps):
        if include_types is not None and key not in include_types:
            return
        if not pts3:
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
        # sides
        for i in range(n_):
            ni = (i + 1) % n_
            a = front[i]
            b = front[ni]
            c = back[ni]
            d = back[i]
            cat["I"] += [a, a]
            cat["J"] += [b, c]
            cat["K"] += [c, d]
        if show_wireframe and key in STRUCTURE_KEYS:
            _add_wire_edges_from_polygon(pts3)

    def _add_mesh_like(mesh_geom, key):
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

    # ------------------------------- collect geom ------------------------------
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
                key = _surface_category(face)
                _add_polygon(pts, key if key in COLOR_MAP else "default")
            # shades on face
            for sh in chain(getattr(face, "outdoor_shades", []) or (), getattr(face, "indoor_shades", []) or ()):
                sg = getattr(sh, "geometry", None)
                if not sg:
                    continue
                raw_s = getattr(sg, "boundary", None) or getattr(sg, "vertices", None) or getattr(sg, "points", None)
                k = "indoor_shade" if getattr(sh, "is_indoor", False) else "outdoor_shade"
                if raw_s:
                    _add_polygon([_pt_to_tuple(p) for p in raw_s], k)
                else:
                    _add_mesh_like(sg, "shade_mesh")
            # apertures
            for ap in getattr(face, "apertures", []) or []:
                # shade(s) attached to aperture
                for sh in chain(getattr(ap, "outdoor_shades", []) or (), getattr(ap, "indoor_shades", []) or ()):
                    sg = getattr(sh, "geometry", None)
                    if not sg:
                        continue
                    raw_s = getattr(sg, "boundary", None) or getattr(sg, "vertices", None) or getattr(sg, "points", None)
                    k = "indoor_shade" if getattr(sh, "is_indoor", False) else "outdoor_shade"
                    if raw_s:
                        _add_polygon([_pt_to_tuple(p) for p in raw_s], k)
                    else:
                        _add_mesh_like(sg, "shade_mesh")
                ag = getattr(ap, "geometry", None)
                if ag is not None:
                    raw_a = getattr(ag, "boundary", None) or getattr(ag, "vertices", None) or getattr(ag, "points", None)
                    cat = _aperture_category(ap, _surface_category(face))
                    if raw_a:
                        _add_extruded_polygon([_pt_to_tuple(p) for p in raw_a], cat, extrude_eps)
                    else:
                        _add_mesh_like(ag, cat)
            # doors
            for dr in getattr(face, "doors", []) or []:
                for sh in chain(getattr(dr, "outdoor_shades", []) or (), getattr(dr, "indoor_shades", []) or ()):
                    sg = getattr(sh, "geometry", None)
                    if not sg:
                        continue
                    raw_s = getattr(sg, "boundary", None) or getattr(sg, "vertices", None) or getattr(sg, "points", None)
                    k = "indoor_shade" if getattr(sh, "is_indoor", False) else "outdoor_shade"
                    if raw_s:
                        _add_polygon([_pt_to_tuple(p) for p in raw_s], k)
                    else:
                        _add_mesh_like(sg, "shade_mesh")
                dg = getattr(dr, "geometry", None)
                if dg is not None:
                    raw_d = getattr(dg, "boundary", None) or getattr(dg, "vertices", None) or getattr(dg, "points", None)
                    cat = _door_category(dr, _surface_category(face))
                    if raw_d:
                        _add_extruded_polygon([_pt_to_tuple(p) for p in raw_d], cat, extrude_eps)
                    else:
                        _add_mesh_like(dg, cat)

    # orphaned shades / shade meshes
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
        if not geom:
            continue
        raw = getattr(geom, "boundary", None) or getattr(geom, "vertices", None) or getattr(geom, "points", None)
        if not raw:
            continue
        pts = [_pt_to_tuple(p) for p in raw]
        fcat = _surface_category(face)
        _add_polygon(pts, fcat if fcat in COLOR_MAP else "default")
        for ap in getattr(face, "apertures", []) or []:
            ag = getattr(ap, "geometry", None)
            if not ag:
                continue
            raw_a = getattr(ag, "boundary", None) or getattr(ag, "vertices", None) or getattr(ag, "points", None)
            cat = _aperture_category(ap, fcat)
            if raw_a:
                _add_extruded_polygon([_pt_to_tuple(p) for p in raw_a], cat, extrude_eps)
            else:
                _add_mesh_like(ag, cat)
        for dr in getattr(face, "doors", []) or []:
            dg = getattr(dr, "geometry", None)
            if not dg:
                continue
            raw_d = getattr(dg, "boundary", None) or getattr(dg, "vertices", None) or getattr(dg, "points", None)
            cat = _door_category(dr, fcat)
            if raw_d:
                _add_extruded_polygon([_pt_to_tuple(p) for p in raw_d], cat, extrude_eps)
            else:
                _add_mesh_like(dg, cat)

    for ap in getattr(model, "orphaned_apertures", []) or []:
        ag = getattr(ap, "geometry", None)
        if ag:
            raw = getattr(ag, "boundary", None) or getattr(ag, "vertices", None) or getattr(ag, "points", None)
            cat = _aperture_category(ap, "exterior_wall")
            if raw:
                _add_extruded_polygon([_pt_to_tuple(p) for p in raw], cat, extrude_eps)
            else:
                _add_mesh_like(ag, cat)

    for dr in getattr(model, "orphaned_doors", []) or []:
        dg = getattr(dr, "geometry", None)
        if dg:
            raw = getattr(dg, "boundary", None) or getattr(dg, "vertices", None) or getattr(dg, "points", None)
            cat = _door_category(dr, "exterior_wall")
            if raw:
                _add_extruded_polygon([_pt_to_tuple(p) for p in raw], cat, extrude_eps)
            else:
                _add_mesh_like(dg, cat)

    # --------------------------- build Plotly figure ---------------------------
    import plotly.graph_objects as go  # local for clarity in notebooks

    fig = go.Figure()
    if include_types is not None:
        include_types = set(include_types)

    draw_order = [
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

    # filled categories in preferred order
    for key in draw_order:
        if include_types is not None and key not in include_types:
            continue
        data = categories.get(key)
        if not data or not data["pts"] or not data["I"]:
            continue
        X, Y, Z = zip(*data["pts"])
        fig.add_trace(
            go.Mesh3d(
                x=X,
                y=Y,
                z=Z,
                i=data["I"],
                j=data["J"],
                k=data["K"],
                color=COLOR_MAP.get(key, COLOR_MAP["default"]),
                opacity=float(np.clip(surface_opacity, 0.0, 1.0)),
                flatshading=True,
                name=key,
                hoverinfo="skip",
                lighting=dict(ambient=0.6, diffuse=0.4),
                showlegend=show_legend,
            )
        )

    # any remaining categories not in draw_order
    for key, data in categories.items():
        if key in draw_order:
            continue
        if include_types is not None and key not in include_types:
            continue
        if not data["pts"] or not data["I"]:
            continue
        X, Y, Z = zip(*data["pts"])
        fig.add_trace(
            go.Mesh3d(
                x=X,
                y=Y,
                z=Z,
                i=data["I"],
                j=data["J"],
                k=data["K"],
                color=COLOR_MAP.get(key, COLOR_MAP["default"]),
                opacity=float(np.clip(surface_opacity, 0.0, 1.0)),
                flatshading=True,
                name=key,
                hoverinfo="skip",
                lighting=dict(ambient=0.6, diffuse=0.4),
                showlegend=show_legend,
            )
        )

    # optional wireframe overlay (structural only)
    if show_wireframe and edge_segments:
        xs, ys, zs = [], [], []
        for (p, q) in edge_segments:
            xs += [p[0], q[0], None]
            ys += [p[1], q[1], None]
            zs += [p[2], q[2], None]
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                name="wireframe",
                line=dict(width=2, color="black"),
                hoverinfo="skip",
                showlegend=show_legend,
            )
        )

    # --- layout & rendering ---
    layout_title = str(title_val) if title_val is not None else None
    fig.update_layout(
        title=layout_title,
        width=int(figsize[0]) if figsize and len(figsize) >= 1 else 640,
        height=int(figsize[1]) if figsize and len(figsize) >= 2 else 420,
        margin=dict(l=0, r=0, t=30 if layout_title else 6, b=0),
        paper_bgcolor=background_color or "white",
        plot_bgcolor=background_color or "white",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode=aspectmode,
        ),
        showlegend=show_legend,
    )

    if show:
        # Show once and return None to avoid Jupyter auto-rendering the return value
        fig.show(config={"scrollZoom": True, "displaylogo": False})
        return None
    else:
        # Don't show; let the caller decide and avoid double-rendering
        return fig

