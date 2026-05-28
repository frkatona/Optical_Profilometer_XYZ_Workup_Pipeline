import argparse
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
from werkzeug.utils import secure_filename

from analyze_heightmap import run_analysis_pipeline


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path(os.environ.get("WEBUI_DATA_DIR", APP_ROOT / "webui_data"))
DEFAULT_MAX_WORKERS = int(os.environ.get("WEBUI_MAX_WORKERS", "2"))
AVAILABLE_EXPORT_MAPS = ["raw", "form", "roughness", "waviness+roughness"]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def blender_executable():
    return shutil.which("blender")


def blender_available():
    return blender_executable() is not None


def sample_available():
    return (APP_ROOT / "test.xyz").exists()


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
        preview_url = None
        if suffix in IMAGE_SUFFIXES:
            preview_url = f"/api/jobs/{job_id}/artifacts/{path.name}?download=0"
        return {
            "name": path.name,
            "label": label,
            "category": category,
            "size_bytes": path.stat().st_size,
            "download_url": f"/api/jobs/{job_id}/artifacts/{path.name}?download=1",
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


def normalize_options(form):
    export_maps = [value for value in form.getlist("export_obj") if value in AVAILABLE_EXPORT_MAPS]
    options = {
        "resolution_factor": int(form.get("resolution_factor", "1")),
        "interpolate": form.get("interpolate", "bilinear"),
        "stats_only": parse_bool(form.get("stats_only")),
        "bounds": parse_bounds(form),
        "export_obj": export_maps,
    }

    render_enabled = parse_bool(form.get("enable_render"))
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
        "camera_elevation": float(form.get("camera_elevation", "32")),
        "camera_lens": float(form.get("camera_lens", "55")),
        "rotation_x": float(form.get("rotation_x", "0")),
        "rotation_y": float(form.get("rotation_y", "0")),
        "rotation_z": float(form.get("rotation_z", "0")),
        "world_strength": float(form.get("world_strength", "0.85")),
        "key_light_energy": float(form.get("key_light_energy", "3500")),
        "fill_light_energy": float(form.get("fill_light_energy", "1.8")),
        "material_color": form.get("material_color", "#c7d3e5"),
        "material_roughness": float(form.get("material_roughness", "0.38")),
        "material_metallic": float(form.get("material_metallic", "0.08")),
        "material_specular": float(form.get("material_specular", "0.55")),
        "transparent_background": parse_bool(form.get("transparent_background")),
    }

    if render_enabled and render_options["render_source"] not in options["export_obj"]:
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
        "--material-color",
        render_options["material_color"],
        "--material-roughness",
        str(render_options["material_roughness"]),
        "--material-metallic",
        str(render_options["material_metallic"]),
        "--material-specular",
        str(render_options["material_specular"]),
    ] + (["--transparent-background"] if render_options["transparent_background"] else [])


def run_blender_render(job_id, input_obj, output_dir, render_options):
    if not blender_available():
        raise RuntimeError("Blender is not available on this server. Use the web-ui Docker target or install Blender.")

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
    )
    assert process.stdout is not None
    for line in process.stdout:
        line = line.strip()
        if line:
            manager.append_log(job_id, line)

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Blender render failed with exit code {return_code}")

    return output_path


def process_job(job_id, input_path):
    job = manager.get(job_id)
    if not job:
        return

    manager.update(job_id, status="running", started_at=time.time(), stage="starting", message="Starting job")
    log_writer = JobLogWriter(lambda line: manager.append_log(job_id, line))
    output_dir = Path(job["paths"]["output_dir"])

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

        artifacts = []
        if result["artifacts"]["statistics_txt"]:
            artifacts.append(manager.serialize_artifact(job_id, result["artifacts"]["statistics_txt"], "Statistics report (TXT)", "stats"))
        if result["artifacts"]["statistics_csv"]:
            artifacts.append(manager.serialize_artifact(job_id, result["artifacts"]["statistics_csv"], "Statistics CSV", "stats"))
        if result["artifacts"]["analysis_png"]:
            artifacts.append(manager.serialize_artifact(job_id, result["artifacts"]["analysis_png"], "Analysis preview", "preview"))
        for label, path in result["artifacts"]["obj"].items():
            artifacts.append(manager.serialize_artifact(job_id, path, f"{label} OBJ", "obj"))

        render_output = None
        if job["render_options"]["enabled"]:
            source_label = job["render_options"]["render_source"]
            input_obj = result["artifacts"]["obj"].get(source_label)
            if not input_obj:
                raise RuntimeError(f"Requested render source '{source_label}' was not generated")
            render_output = run_blender_render(job_id, input_obj, output_dir, job["render_options"])
            artifacts.append(manager.serialize_artifact(job_id, render_output, "Rendered image", "render"))

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
        )


def job_payload(job):
    if not job:
        return None
    payload = dict(job)
    payload["stats_summary"] = summarize_stats(job.get("stats"))
    return payload


@app.get("/")
def index():
    return render_template(
        "index.html",
        blender_available=blender_available(),
        sample_available=sample_available(),
        max_workers=DEFAULT_MAX_WORKERS,
        export_maps=AVAILABLE_EXPORT_MAPS,
    )


@app.get("/api/capabilities")
def capabilities():
    return jsonify(
        {
            "blender_available": blender_available(),
            "sample_available": sample_available(),
            "max_workers": DEFAULT_MAX_WORKERS,
            "export_maps": AVAILABLE_EXPORT_MAPS,
        }
    )


@app.get("/api/jobs")
def list_jobs():
    return jsonify({"jobs": [job_payload(job) for job in manager.list()]})


@app.post("/api/jobs")
def create_job():
    use_sample = parse_bool(request.form.get("use_bundled_sample"))
    upload = request.files.get("input_file")

    if not use_sample and (upload is None or not upload.filename):
        return jsonify({"error": "Upload a .xyz file or use the bundled sample file."}), 400

    try:
        options, render_options = normalize_options(request.form)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if render_options["enabled"] and not blender_available():
        return jsonify({"error": "Blender rendering is not available on this server."}), 400

    source_name = "test.xyz" if use_sample else secure_filename(upload.filename)
    job = manager.create_job(source_name, options, render_options)

    input_dir = Path(job["paths"]["input_dir"])
    if use_sample:
        sample_path = APP_ROOT / "test.xyz"
        if not sample_path.exists():
            return jsonify({"error": "Bundled sample test.xyz is not available on this server."}), 400
        input_path = input_dir / sample_path.name
        shutil.copy2(sample_path, input_path)
    else:
        filename = secure_filename(upload.filename) or "input.xyz"
        if not filename.lower().endswith(".xyz"):
            filename += ".xyz"
        input_path = input_dir / filename
        upload.save(input_path)

    manager.submit(process_job, job["id"], input_path)
    return jsonify({"job": job_payload(manager.get(job["id"]))}), 202


@app.get("/api/jobs/<job_id>")
def get_job(job_id):
    job = manager.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({"job": job_payload(job)})


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
