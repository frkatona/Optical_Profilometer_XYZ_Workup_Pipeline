import argparse
import base64
import io
import os
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

from flask import Flask, abort, jsonify, render_template, request, send_file
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from werkzeug.utils import secure_filename

from analyze_heightmap import inspect_xyz_layout, iter_xyz_records, run_analysis_pipeline


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path(os.environ.get("WEBUI_DATA_DIR", APP_ROOT / "webui_data"))
DEFAULT_MAX_WORKERS = int(os.environ.get("WEBUI_MAX_WORKERS", "2"))
AVAILABLE_EXPORT_MAPS = ["raw", "form", "roughness", "waviness+roughness"]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PREVIEW_MAX_DIMENSION = 128
PREVIEW_NO_DATA_COLOR = "#ff5b73"
SURFACE_MESH_SOURCE_MAX_AXIS = 240
SURFACE_MESH_RENDER_MAX_AXIS = 170
SURFACE_MESH_BASE_AXIS = 88
SURFACE_MESH_MAX_VERTICES = 60000
BENIGN_WORKBENCH_LINES = {
    "EGL Error (0x3009): EGL_BAD_MATCH: Arguments are inconsistent (for example, a valid context requires buffers not supplied by a valid surface)."
}


def blender_executable():
    return shutil.which("blender")


def blender_available():
    return blender_executable() is not None


def preview_resolution_factor(width, height):
    largest_dimension = max(width, height)
    if largest_dimension <= PREVIEW_MAX_DIMENSION:
        return 1
    return max(1, (largest_dimension + PREVIEW_MAX_DIMENSION - 1) // PREVIEW_MAX_DIMENSION)


def load_xyz_preview(filepath):
    layout = inspect_xyz_layout(filepath)
    width = layout["width"]
    height = layout["height"]
    factor = preview_resolution_factor(width, height)
    preview_width = max(1, (width + factor - 1) // factor)
    preview_height = max(1, (height + factor - 1) // factor)
    sums = np.zeros((preview_height, preview_width), dtype=float)
    counts = np.zeros((preview_height, preview_width), dtype=np.uint32)

    for raw_x, raw_y, z in iter_xyz_records(filepath, layout["data_start_line"]):
        if z is None:
            continue

        if layout["x_index"] is not None and layout["y_index"] is not None:
            x = layout["x_index"].get(raw_x)
            y = layout["y_index"].get(raw_y)
            if x is None or y is None:
                continue
        else:
            x = int(raw_x)
            y = int(raw_y)

        preview_x = x // factor
        preview_y = y // factor
        if 0 <= preview_x < preview_width and 0 <= preview_y < preview_height:
            sums[preview_y, preview_x] += z
            counts[preview_y, preview_x] += 1

    data = np.full((preview_height, preview_width), np.nan, dtype=float)
    np.divide(sums, counts, out=data, where=counts > 0)
    return data, {
        "width": width,
        "height": height,
        "preview_width": preview_width,
        "preview_height": preview_height,
        "resolution_factor": factor,
        "sampled_points": int(counts.sum()),
        "pixel_spacing_um": layout["pixel_spacing_um"],
        "warnings": layout["warnings"],
        "header_present": layout["header_present"],
        "coordinate_mode": layout["coordinate_mode"],
    }


def render_preview_payload(filepath, filename):
    data, metadata = load_xyz_preview(filepath)
    finite = np.isfinite(data)
    if not finite.any():
        raise ValueError("The selected XYZ file did not contain any valid height values.")

    values = data[finite]
    vmin, vmax = np.percentile(values, [2, 98])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if vmin == vmax:
            vmax = vmin + 1.0

    fig = Figure(figsize=(1.28, 1.28), dpi=100)
    fig.patch.set_facecolor("#f6f9fc")
    axis = fig.subplots()
    cmap = colormaps["viridis"].copy()
    cmap.set_bad(PREVIEW_NO_DATA_COLOR)
    axis.imshow(data, cmap=cmap, interpolation="nearest", aspect="auto", vmin=vmin, vmax=vmax)
    axis.set_axis_off()
    fig.tight_layout(pad=0)

    buffer = io.BytesIO()
    FigureCanvas(fig).print_png(buffer)
    image_data = base64.b64encode(buffer.getvalue()).decode("ascii")
    coverage = float(np.count_nonzero(finite) / data.size * 100.0)

    return {
        "filename": filename,
        "image_url": f"data:image/png;base64,{image_data}",
        "width": metadata["width"],
        "height": metadata["height"],
        "preview_width": metadata["preview_width"],
        "preview_height": metadata["preview_height"],
        "resolution_factor": metadata["resolution_factor"],
        "sampled_points": metadata["sampled_points"],
        "pixel_spacing_um": metadata["pixel_spacing_um"],
        "coverage_percent": round(coverage, 1),
        "no_data_percent": round(100.0 - coverage, 1),
        "no_data_color": PREVIEW_NO_DATA_COLOR,
        "z_min": float(np.min(values)),
        "z_max": float(np.max(values)),
        "warnings": metadata["warnings"],
        "header_present": metadata["header_present"],
        "coordinate_mode": metadata["coordinate_mode"],
    }


def choose_surface_mesh_factor(width, height, bounds=None, pixel_spacing_um=1.0):
    if bounds:
        x1_um, x2_um, y1_um, y2_um = bounds
        width = max(1, int(abs(x2_um - x1_um) / max(pixel_spacing_um, 1e-9)))
        height = max(1, int(abs(y2_um - y1_um) / max(pixel_spacing_um, 1e-9)))
    largest_dimension = max(width, height)
    if largest_dimension <= SURFACE_MESH_SOURCE_MAX_AXIS:
        return 1
    return max(1, (largest_dimension + SURFACE_MESH_SOURCE_MAX_AXIS - 1) // SURFACE_MESH_SOURCE_MAX_AXIS)


def fill_heightmap_nans(data):
    finite = np.isfinite(data)
    if not finite.any():
        raise ValueError("The selected job does not contain any valid height values for 3D viewing.")
    fill_value = float(np.nanmedian(data))
    return np.where(finite, data, fill_value), finite


def load_surface_mesh_grid(filepath, bounds=None):
    layout = inspect_xyz_layout(filepath)
    width = layout["width"]
    height = layout["height"]
    source_spacing_um = layout["pixel_spacing_um"]
    factor = choose_surface_mesh_factor(width, height, bounds=bounds, pixel_spacing_um=source_spacing_um)

    x_start, x_stop = 0, width
    y_start, y_stop = 0, height
    if bounds:
        x1_um, x2_um, y1_um, y2_um = bounds
        x_start = max(0, min(width - 1, int(min(x1_um, x2_um) / max(source_spacing_um, 1e-9))))
        x_stop = max(x_start + 1, min(width, int(max(x1_um, x2_um) / max(source_spacing_um, 1e-9))))
        y_start = max(0, min(height - 1, int(min(y1_um, y2_um) / max(source_spacing_um, 1e-9))))
        y_stop = max(y_start + 1, min(height, int(max(y1_um, y2_um) / max(source_spacing_um, 1e-9))))

    mesh_width = max(1, (x_stop - x_start + factor - 1) // factor)
    mesh_height = max(1, (y_stop - y_start + factor - 1) // factor)
    sums = np.zeros((mesh_height, mesh_width), dtype=float)
    counts = np.zeros((mesh_height, mesh_width), dtype=np.uint32)

    for raw_x, raw_y, z in iter_xyz_records(filepath, layout["data_start_line"]):
        if z is None:
            continue

        if layout["x_index"] is not None and layout["y_index"] is not None:
            x = layout["x_index"].get(raw_x)
            y = layout["y_index"].get(raw_y)
            if x is None or y is None:
                continue
        else:
            x = int(raw_x)
            y = int(raw_y)

        if not (x_start <= x < x_stop and y_start <= y < y_stop):
            continue

        mesh_x = (x - x_start) // factor
        mesh_y = (y - y_start) // factor
        if 0 <= mesh_x < mesh_width and 0 <= mesh_y < mesh_height:
            sums[mesh_y, mesh_x] += z
            counts[mesh_y, mesh_x] += 1

    data = np.full((mesh_height, mesh_width), np.nan, dtype=float)
    np.divide(sums, counts, out=data, where=counts > 0)
    return data, counts > 0, {
        "width": width,
        "height": height,
        "mesh_width": mesh_width,
        "mesh_height": mesh_height,
        "sample_factor": factor,
        "pixel_spacing_um": source_spacing_um * factor,
        "source_pixel_spacing_um": source_spacing_um,
        "warnings": layout["warnings"],
        "header_present": layout["header_present"],
        "coordinate_mode": layout["coordinate_mode"],
        "bounds_px": [x_start, x_stop, y_start, y_stop],
    }


def adaptive_axis_indices(importance, axis, max_count):
    length = importance.shape[axis]
    if length <= max_count:
        return np.arange(length, dtype=int)

    base_count = min(length, max(16, min(SURFACE_MESH_BASE_AXIS, int(max_count * 0.6))))
    base = set(np.linspace(0, length - 1, base_count, dtype=int).tolist())
    scores = np.nanmax(importance, axis=1 if axis == 0 else 0)
    extra_count = max(0, max_count - len(base))

    for index in np.argsort(scores)[::-1]:
        for candidate in (int(index), int(index) - 1, int(index) + 1):
            if 0 <= candidate < length:
                base.add(candidate)
            if len(base) >= max_count:
                break
        if len(base) >= max_count:
            break

    return np.array(sorted(base), dtype=int)


def select_adaptive_surface_samples(data):
    max_axis = min(SURFACE_MESH_RENDER_MAX_AXIS, int(np.sqrt(SURFACE_MESH_MAX_VERTICES)))
    if data.shape[0] * data.shape[1] <= SURFACE_MESH_MAX_VERTICES and max(data.shape) <= max_axis:
        return np.arange(data.shape[0], dtype=int), np.arange(data.shape[1], dtype=int), "native"

    filled, _ = fill_heightmap_nans(data)
    grad_y, grad_x = np.gradient(filled)
    importance = np.sqrt((grad_x ** 2) + (grad_y ** 2))

    rows = adaptive_axis_indices(importance, axis=0, max_count=min(max_axis, data.shape[0]))
    cols = adaptive_axis_indices(importance, axis=1, max_count=min(max_axis, data.shape[1]))
    while len(rows) * len(cols) > SURFACE_MESH_MAX_VERTICES:
        if len(rows) >= len(cols) and len(rows) > 2:
            rows = rows[::2]
        elif len(cols) > 2:
            cols = cols[::2]
        else:
            break
    return rows, cols, "adaptive-gradient"


def build_surface_mesh_payload(filepath, filename, bounds=None):
    data, valid_mask, metadata = load_surface_mesh_grid(filepath, bounds=bounds)
    filled, finite = fill_heightmap_nans(data)
    rows, cols, remesh_mode = select_adaptive_surface_samples(filled)

    selected = filled[np.ix_(rows, cols)]
    selected_valid = valid_mask[np.ix_(rows, cols)]
    height_values = selected[selected_valid] if selected_valid.any() else selected.ravel()
    height_min = float(np.min(height_values))
    height_max = float(np.max(height_values))
    height_mid = float(np.median(height_values))
    height_range = max(height_max - height_min, 1e-9)
    spacing_um = metadata["pixel_spacing_um"]
    lateral_width_um = max((cols[-1] - cols[0]) * spacing_um, spacing_um)
    lateral_height_um = max((rows[-1] - rows[0]) * spacing_um, spacing_um)
    lateral_span_um = max(lateral_width_um, lateral_height_um, spacing_um)
    relief_scale = (lateral_span_um * 0.18) / height_range

    vertices = []
    values = []
    for row in rows:
        y_um = row * spacing_um - lateral_height_um / 2
        for col in cols:
            x_um = col * spacing_um - lateral_width_um / 2
            z_um = (filled[row, col] - height_mid) * relief_scale
            vertices.extend([
                x_um / lateral_span_um,
                z_um / lateral_span_um,
                y_um / lateral_span_um,
            ])
            values.append((filled[row, col] - height_min) / height_range)

    row_count = len(rows)
    col_count = len(cols)
    indices = []
    for row_index in range(row_count - 1):
        for col_index in range(col_count - 1):
            v00 = row_index * col_count + col_index
            v01 = v00 + 1
            v10 = (row_index + 1) * col_count + col_index
            v11 = v10 + 1
            valid00 = selected_valid[row_index, col_index]
            valid01 = selected_valid[row_index, col_index + 1]
            valid10 = selected_valid[row_index + 1, col_index]
            valid11 = selected_valid[row_index + 1, col_index + 1]
            if valid00 and valid10 and valid01:
                indices.extend([v00, v10, v01])
            if valid01 and valid10 and valid11:
                indices.extend([v01, v10, v11])

    if not indices:
        for row_index in range(row_count - 1):
            for col_index in range(col_count - 1):
                v00 = row_index * col_count + col_index
                v01 = v00 + 1
                v10 = (row_index + 1) * col_count + col_index
                v11 = v10 + 1
                indices.extend([v00, v10, v01, v01, v10, v11])

    vertex_array = np.array(vertices, dtype=np.float32).reshape((-1, 3))
    normals = np.zeros_like(vertex_array)
    for tri_start in range(0, len(indices), 3):
        a, b, c = indices[tri_start:tri_start + 3]
        normal = np.cross(vertex_array[b] - vertex_array[a], vertex_array[c] - vertex_array[a])
        length = np.linalg.norm(normal)
        if length > 1e-9:
            normal = normal / length
        normals[a] += normal
        normals[b] += normal
        normals[c] += normal

    normal_lengths = np.linalg.norm(normals, axis=1)
    normals[normal_lengths > 1e-9] /= normal_lengths[normal_lengths > 1e-9, None]
    normals[normal_lengths <= 1e-9] = [0.0, 1.0, 0.0]

    return {
        "filename": filename,
        "vertices": [round(float(value), 6) for value in vertex_array.ravel()],
        "normals": [round(float(value), 6) for value in normals.ravel()],
        "values": [round(float(value), 6) for value in values],
        "indices": indices,
        "metadata": {
            **metadata,
            "selected_rows": int(row_count),
            "selected_cols": int(col_count),
            "vertex_count": int(vertex_array.shape[0]),
            "triangle_count": int(len(indices) // 3),
            "remesh_mode": remesh_mode,
            "height_min_um": height_min,
            "height_max_um": height_max,
            "height_exaggeration": relief_scale,
            "valid_coverage_percent": round(float(np.count_nonzero(finite) / finite.size * 100.0), 2),
        },
    }


class JobLogWriter(io.TextIOBase):
    def __init__(self, callback):
        self.callback = callback
        self.buffer = ""

    def write(self, text):
        if not text:
            return 0
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            line = line.strip()
            if line:
                self.callback(line)
        return len(text)

    def flush(self):
        line = self.buffer.strip()
        if line:
            self.callback(line)
        self.buffer = ""


class JobManager:
    def __init__(self, base_dir, max_workers):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.lock = threading.Lock()
        self.jobs = {}

    def _job_dir(self, job_id):
        return self.base_dir / job_id

    def create_job(self, source_name, options, render_options):
        job_id = uuid.uuid4().hex[:12]
        job_dir = self._job_dir(job_id)
        input_dir = job_dir / "input"
        output_dir = job_dir / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        record = {
            "id": job_id,
            "source_name": source_name,
            "status": "queued",
            "stage": "queued",
            "message": "Waiting to start",
            "progress": 0.0,
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "logs": [],
            "options": options,
            "render_options": render_options,
            "stats": None,
            "metadata": None,
            "artifacts": [],
            "paths": {
                "job_dir": str(job_dir),
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
            },
        }
        with self.lock:
            self.jobs[job_id] = record
        return record

    def get(self, job_id):
        with self.lock:
            record = self.jobs.get(job_id)
            return dict(record) if record else None

    def list(self):
        with self.lock:
            records = [dict(job) for job in self.jobs.values()]
        return sorted(records, key=lambda item: item["created_at"], reverse=True)

    def update(self, job_id, **fields):
        with self.lock:
            record = self.jobs[job_id]
            record.update(fields)

    def append_log(self, job_id, message):
        lines = [line.strip() for line in str(message).replace("\r", "\n").split("\n")]
        lines = [line for line in lines if line]
        if not lines:
            return
        with self.lock:
            record = self.jobs[job_id]
            record["logs"].extend(lines)
            record["logs"] = record["logs"][-300:]

    def serialize_artifact(self, job_id, path, label, category):
        path = Path(path)
        if not path.exists():
            return None
        suffix = path.suffix.lower()
        version_token = str(path.stat().st_mtime_ns)
        preview_url = None
        if suffix in IMAGE_SUFFIXES:
            preview_url = f"/api/jobs/{job_id}/artifacts/{path.name}?download=0&v={version_token}"
        return {
            "name": path.name,
            "label": label,
            "category": category,
            "size_bytes": path.stat().st_size,
            "download_url": f"/api/jobs/{job_id}/artifacts/{path.name}?download=1&v={version_token}",
            "preview_url": preview_url,
        }

    def submit(self, fn, *args):
        self.executor.submit(fn, *args)


app = Flask(__name__, template_folder="templates", static_folder="static")
manager = JobManager(DEFAULT_DATA_DIR, DEFAULT_MAX_WORKERS)


def parse_bool(value):
    return str(value).lower() in {"1", "true", "on", "yes"}


def parse_float(value, field_name):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a number") from exc


def parse_int(value, field_name):
    if value in (None, ""):
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an integer") from exc


def parse_bounds(form):
    values = [form.get("bound_x1"), form.get("bound_x2"), form.get("bound_y1"), form.get("bound_y2")]
    if not any(values):
        return None
    if not all(values):
        raise ValueError("Bounds require all four values")
    return tuple(float(value) for value in values)


def normalize_render_options(form, enabled=None):
    render_enabled = parse_bool(form.get("enable_render")) if enabled is None else bool(enabled)
    render_options = {
        "enabled": render_enabled,
        "render_source": form.get("render_source", "roughness"),
        "output_format": form.get("render_output_format", "png"),
        "engine": form.get("render_engine", "CYCLES"),
        "resolution_x": int(form.get("render_resolution_x", "1600")),
        "resolution_y": int(form.get("render_resolution_y", "1200")),
        "samples": int(form.get("render_samples", "64")),
        "camera_distance_scale": float(form.get("camera_distance_scale", "2.1")),
        "camera_azimuth": float(form.get("camera_azimuth", "35")),
        "camera_elevation": float(form.get("camera_elevation", "55")),
        "camera_lens": float(form.get("camera_lens", "75")),
        "height_scale": float(form.get("height_scale", "150")),
        "auto_height_scale": parse_bool(form.get("auto_height_scale")),
        "auto_height_ratio": float(form.get("auto_height_ratio", "0.12")),
        "rotation_x": float(form.get("rotation_x", "-10")),
        "rotation_y": float(form.get("rotation_y", "-80")),
        "rotation_z": float(form.get("rotation_z", "-90")),
        "world_strength": float(form.get("world_strength", "0.85")),
        "key_light_energy": float(form.get("key_light_energy", "3500")),
        "fill_light_energy": float(form.get("fill_light_energy", "1.8")),
        "shading": form.get("shading", "smooth"),
        "material_preset": form.get("material_preset", "custom"),
        "material_color": form.get("material_color", "#c7d3e5"),
        "material_roughness": float(form.get("material_roughness", "0.38")),
        "material_metallic": float(form.get("material_metallic", "0.08")),
        "material_specular": float(form.get("material_specular", "0.55")),
        "transparent_background": parse_bool(form.get("transparent_background")),
    }

    if render_options["height_scale"] <= 0:
        raise ValueError("Height scale must be greater than 0")
    if render_options["auto_height_ratio"] <= 0:
        raise ValueError("Auto height ratio must be greater than 0")
    return render_options


def normalize_options(form):
    export_maps = [value for value in form.getlist("export_obj") if value in AVAILABLE_EXPORT_MAPS]
    options = {
        "resolution_factor": int(form.get("resolution_factor", "1")),
        "interpolate": form.get("interpolate", "bilinear"),
        "stats_only": parse_bool(form.get("stats_only")),
        "bounds": parse_bounds(form),
        "export_obj": export_maps,
    }

    render_options = normalize_render_options(form)

    if render_options["enabled"] and render_options["render_source"] not in options["export_obj"]:
        options["export_obj"] = list(dict.fromkeys(options["export_obj"] + [render_options["render_source"]]))

    return options, render_options


def summarize_stats(stats):
    if not stats:
        return None
    keys = [
        "coverage_percent",
        "min",
        "max",
        "mean",
        "median",
        "std",
        "Ra",
        "Rq",
        "Rz",
    ]
    return {key: stats.get(key) for key in keys}


def build_analysis_artifacts(job_id, result):
    artifacts = []
    if result["artifacts"]["statistics_txt"]:
        artifacts.append(
            manager.serialize_artifact(job_id, result["artifacts"]["statistics_txt"], "Statistics report (TXT)", "stats")
        )
    if result["artifacts"]["statistics_csv"]:
        artifacts.append(
            manager.serialize_artifact(job_id, result["artifacts"]["statistics_csv"], "Statistics CSV", "stats")
        )
    if result["artifacts"]["analysis_png"]:
        artifacts.append(
            manager.serialize_artifact(job_id, result["artifacts"]["analysis_png"], "Analysis preview", "preview")
        )
    for label, path in result["artifacts"]["obj"].items():
        artifacts.append(manager.serialize_artifact(job_id, path, f"{label} OBJ", "obj"))
    return [artifact for artifact in artifacts if artifact]


def available_obj_sources(job):
    output_dir = Path(job["paths"]["output_dir"])
    return [source for source in AVAILABLE_EXPORT_MAPS if any(output_dir.glob(f"*_{source}.obj"))]


def find_obj_path(job, source_label):
    output_dir = Path(job["paths"]["output_dir"])
    matches = sorted(output_dir.glob(f"*_{source_label}.obj"))
    return matches[0] if matches else None


def find_input_xyz_path(job):
    input_dir = Path(job["paths"]["input_dir"])
    matches = sorted(input_dir.glob("*.xyz"))
    return matches[0] if matches else None


def without_artifact_category(artifacts, category):
    return [artifact for artifact in artifacts if artifact.get("category") != category]


def cleanup_old_render_outputs(job, keep_path):
    output_dir = Path(job["paths"]["output_dir"])
    keep_path = Path(keep_path)
    for path in sorted(output_dir.glob("*_render.*")):
        if path == keep_path:
            continue
        path.unlink(missing_ok=True)


def render_artifact_label(source_label):
    return f"Rendered image ({source_label})"


def build_blender_command(input_obj, output_image, render_options):
    script_path = APP_ROOT / "blender" / "render_obj.py"
    return [
        blender_executable(),
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        str(script_path),
        "--",
        "--input",
        str(input_obj),
        "--output",
        str(output_image),
        "--engine",
        render_options["engine"],
        "--resolution-x",
        str(render_options["resolution_x"]),
        "--resolution-y",
        str(render_options["resolution_y"]),
        "--samples",
        str(render_options["samples"]),
        "--camera-distance-scale",
        str(render_options["camera_distance_scale"]),
        "--camera-azimuth",
        str(render_options["camera_azimuth"]),
        "--camera-elevation",
        str(render_options["camera_elevation"]),
        "--camera-lens",
        str(render_options["camera_lens"]),
        "--height-scale",
        str(render_options["height_scale"]),
        "--auto-height-ratio",
        str(render_options["auto_height_ratio"]),
        "--rotation-x",
        str(render_options["rotation_x"]),
        "--rotation-y",
        str(render_options["rotation_y"]),
        "--rotation-z",
        str(render_options["rotation_z"]),
        "--world-strength",
        str(render_options["world_strength"]),
        "--key-light-energy",
        str(render_options["key_light_energy"]),
        "--fill-light-energy",
        str(render_options["fill_light_energy"]),
        "--shading",
        render_options["shading"],
        "--material-preset",
        render_options["material_preset"],
        "--material-color",
        render_options["material_color"],
        "--material-roughness",
        str(render_options["material_roughness"]),
        "--material-metallic",
        str(render_options["material_metallic"]),
        "--material-specular",
        str(render_options["material_specular"]),
    ] + (["--auto-height-scale"] if render_options["auto_height_scale"] else []) + (
        ["--transparent-background"] if render_options["transparent_background"] else []
    )


def run_blender_render(job_id, input_obj, output_dir, render_options):
    if not blender_available():
        raise RuntimeError("Blender is not available on this server. Use the blender Docker target or install Blender.")

    output_name = f"{Path(input_obj).stem}_render.{render_options['output_format']}"
    output_path = Path(output_dir) / output_name
    command = build_blender_command(input_obj, output_path, render_options)
    manager.update(job_id, progress=90.0, stage="rendering", message="Running Blender render")
    manager.append_log(job_id, "Starting Blender render...")

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={
            **os.environ,
            "LIBGL_ALWAYS_SOFTWARE": os.environ.get("LIBGL_ALWAYS_SOFTWARE", "1"),
        },
    )
    assert process.stdout is not None
    deferred_lines = []
    for line in process.stdout:
        line = line.strip()
        if line:
            if line in BENIGN_WORKBENCH_LINES:
                deferred_lines.append(line)
            else:
                manager.append_log(job_id, line)

    return_code = process.wait()
    if return_code != 0:
        for line in deferred_lines:
            manager.append_log(job_id, line)
        raise RuntimeError(f"Blender render failed with exit code {return_code}")

    return output_path


def process_rerender(job_id, render_options):
    job = manager.get(job_id)
    if not job:
        return

    source_label = render_options["render_source"]
    input_obj = find_obj_path(job, source_label)
    if not input_obj:
        available_sources = ", ".join(available_obj_sources(job)) or "none"
        message = (
            f"Requested render source '{source_label}' is not available for this job. "
            f"Available OBJ sources: {available_sources}."
        )
        manager.append_log(job_id, message)
        manager.update(
            job_id,
            status="failed",
            stage="failed",
            message=message,
            error=message,
            finished_at=time.time(),
            render_options=render_options,
        )
        return

    manager.update(
        job_id,
        status="running",
        stage="rerendering",
        message=f"Rerendering existing {source_label} OBJ",
        progress=88.0,
        error=None,
        finished_at=None,
        render_options=render_options,
    )
    manager.append_log(job_id, f"Render-only rerun queued using existing OBJ: {input_obj.name}")

    try:
        output_dir = Path(job["paths"]["output_dir"])
        render_output = run_blender_render(job_id, input_obj, output_dir, render_options)
        cleanup_old_render_outputs(job, render_output)

        latest_job = manager.get(job_id) or job
        artifacts = without_artifact_category(latest_job.get("artifacts", []), "render")
        artifacts.append(
            manager.serialize_artifact(job_id, render_output, render_artifact_label(source_label), "render")
        )
        artifacts = [artifact for artifact in artifacts if artifact]

        manager.append_log(job_id, "Render-only rerun complete.")
        manager.update(
            job_id,
            status="completed",
            stage="completed",
            message="Render-only rerun finished",
            progress=100.0,
            finished_at=time.time(),
            error=None,
            artifacts=artifacts,
            render_options=render_options,
        )
    except Exception as exc:
        manager.append_log(job_id, traceback.format_exc())
        manager.update(
            job_id,
            status="failed",
            stage="failed",
            message=str(exc),
            error=str(exc),
            finished_at=time.time(),
            render_options=render_options,
        )


def process_job(job_id, input_path):
    job = manager.get(job_id)
    if not job:
        return

    manager.update(job_id, status="running", started_at=time.time(), stage="starting", message="Starting job")
    log_writer = JobLogWriter(lambda line: manager.append_log(job_id, line))
    output_dir = Path(job["paths"]["output_dir"])
    result = None
    artifacts = []

    def progress_callback(progress, stage, message):
        manager.update(job_id, progress=round(progress * 100, 1), stage=stage, message=message)

    try:
        with redirect_stdout(log_writer), redirect_stderr(log_writer):
            result = run_analysis_pipeline(
                input_path,
                resolution_factor=job["options"]["resolution_factor"],
                interpolate=job["options"]["interpolate"],
                export_obj=job["options"]["export_obj"],
                output_dir=output_dir,
                no_display=True,
                stats_only=job["options"]["stats_only"],
                bounds=job["options"]["bounds"],
                progress_callback=progress_callback,
                show_progress=False,
            )

        artifacts = build_analysis_artifacts(job_id, result)
        manager.update(
            job_id,
            stats=result["stats"],
            metadata=result["metadata"],
            artifacts=artifacts,
        )

        render_output = None
        if job["render_options"]["enabled"]:
            source_label = job["render_options"]["render_source"]
            input_obj = result["artifacts"]["obj"].get(source_label)
            if not input_obj:
                raise RuntimeError(f"Requested render source '{source_label}' was not generated")
            render_output = run_blender_render(job_id, input_obj, output_dir, job["render_options"])
            artifacts.append(
                manager.serialize_artifact(job_id, render_output, render_artifact_label(source_label), "render")
            )

        artifacts = [artifact for artifact in artifacts if artifact]
        manager.update(
            job_id,
            status="completed",
            progress=100.0,
            stage="completed",
            message="Job finished",
            finished_at=time.time(),
            stats=result["stats"],
            metadata=result["metadata"],
            artifacts=artifacts,
        )
    except Exception as exc:
        manager.append_log(job_id, traceback.format_exc())
        manager.update(
            job_id,
            status="failed",
            stage="failed",
            message=str(exc),
            error=str(exc),
            finished_at=time.time(),
            stats=result["stats"] if result else None,
            metadata=result["metadata"] if result else None,
            artifacts=artifacts,
        )


def job_payload(job):
    if not job:
        return None
    payload = dict(job)
    payload["stats_summary"] = summarize_stats(job.get("stats"))
    payload["available_obj_sources"] = available_obj_sources(job)
    payload["rerender_ready"] = (
        payload["status"] not in {"queued", "running"} and bool(payload["available_obj_sources"])
    )
    return payload


@app.get("/")
def index():
    return render_template(
        "index.html",
        blender_available=blender_available(),
        max_workers=DEFAULT_MAX_WORKERS,
        export_maps=AVAILABLE_EXPORT_MAPS,
    )


@app.get("/api/capabilities")
def capabilities():
    return jsonify(
        {
            "blender_available": blender_available(),
            "max_workers": DEFAULT_MAX_WORKERS,
            "export_maps": AVAILABLE_EXPORT_MAPS,
        }
    )


@app.post("/api/preview")
def preview_input_file():
    upload = request.files.get("input_file")
    if upload is None or not upload.filename:
        return jsonify({"error": "Upload a .xyz file to preview."}), 400

    filename = secure_filename(upload.filename) or "input.xyz"
    if not filename.lower().endswith(".xyz"):
        return jsonify({"error": "Preview requires a .xyz file."}), 400

    preview_dir = DEFAULT_DATA_DIR / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    temp_path = preview_dir / f"{uuid.uuid4().hex}_{filename}"

    try:
        upload.save(temp_path)
        preview = render_preview_payload(temp_path, filename)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Could not render a preview for this XYZ file."}), 400
    finally:
        temp_path.unlink(missing_ok=True)

    return jsonify({"preview": preview})


@app.get("/api/jobs")
def list_jobs():
    return jsonify({"jobs": [job_payload(job) for job in manager.list()]})


@app.post("/api/jobs")
def create_job():
    upload = request.files.get("input_file")

    if upload is None or not upload.filename:
        return jsonify({"error": "Upload a .xyz file."}), 400

    try:
        options, render_options = normalize_options(request.form)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if render_options["enabled"] and not blender_available():
        return jsonify({"error": "Blender rendering is not available on this server."}), 400

    source_name = secure_filename(upload.filename) or "input.xyz"
    job = manager.create_job(source_name, options, render_options)

    input_dir = Path(job["paths"]["input_dir"])
    filename = secure_filename(upload.filename) or "input.xyz"
    if not filename.lower().endswith(".xyz"):
        filename += ".xyz"
    input_path = input_dir / filename
    upload.save(input_path)

    manager.submit(process_job, job["id"], input_path)
    return jsonify({"job": job_payload(manager.get(job["id"]))}), 202


@app.post("/api/jobs/<job_id>/rerender")
def rerender_job(job_id):
    job = manager.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] in {"queued", "running"}:
        return jsonify({"error": "Wait for the current job activity to finish before rerendering."}), 409
    if not blender_available():
        return jsonify({"error": "Blender rendering is not available on this server."}), 400

    try:
        render_options = normalize_render_options(request.form, enabled=True)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if not find_obj_path(job, render_options["render_source"]):
        available_sources = ", ".join(available_obj_sources(job)) or "none"
        return jsonify(
            {
                "error": (
                    f"Selected job does not contain an OBJ for '{render_options['render_source']}'. "
                    f"Available OBJ sources: {available_sources}."
                )
            }
        ), 400

    manager.submit(process_rerender, job_id, render_options)
    manager.update(
        job_id,
        status="queued",
        stage="queued",
        message=f"Queued render-only rerun for {render_options['render_source']}",
        progress=87.0,
        error=None,
        render_options=render_options,
    )
    return jsonify({"job": job_payload(manager.get(job_id))}), 202


@app.get("/api/jobs/<job_id>")
def get_job(job_id):
    job = manager.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({"job": job_payload(job)})


@app.get("/api/jobs/<job_id>/surface-mesh")
def get_surface_mesh(job_id):
    job = manager.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] != "completed":
        return jsonify({"error": "3D surface viewing is available after analysis completes."}), 409

    input_path = find_input_xyz_path(job)
    if not input_path:
        return jsonify({"error": "The original XYZ input file is no longer available for this job."}), 404

    try:
        payload = build_surface_mesh_payload(
            input_path,
            job.get("source_name") or input_path.name,
            bounds=job.get("options", {}).get("bounds"),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Could not build the experimental 3D surface mesh."}), 500

    return jsonify({"mesh": payload})


@app.get("/api/jobs/<job_id>/artifacts/<path:filename>")
def download_artifact(job_id, filename):
    job = manager.get(job_id)
    if not job:
        abort(404)
    output_path = Path(job["paths"]["output_dir"]) / filename
    if not output_path.exists():
        abort(404)
    as_attachment = request.args.get("download", "1") != "0"
    return send_file(output_path, as_attachment=as_attachment, download_name=output_path.name)


def create_app():
    return app


def main():
    parser = argparse.ArgumentParser(description="Run the optical profilometry web UI")
    parser.add_argument("--host", default=os.environ.get("WEBUI_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("WEBUI_PORT", "8000")))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
