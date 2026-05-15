import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Import a Wavefront OBJ surface and render it headlessly."
    )
    parser.add_argument("--input", required=True, help="Path to the input OBJ file")
    parser.add_argument(
        "--output", required=True, help="Path to the rendered output image"
    )
    parser.add_argument(
        "--engine",
        default="CYCLES",
        choices=["CYCLES", "BLENDER_WORKBENCH"],
        help="Blender render engine to use",
    )
    parser.add_argument(
        "--resolution-x", type=int, default=1600, help="Output image width in pixels"
    )
    parser.add_argument(
        "--resolution-y", type=int, default=1200, help="Output image height in pixels"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=64,
        help="Cycles sample count (ignored by non-Cycles engines)",
    )
    parser.add_argument(
        "--camera-distance-scale",
        type=float,
        default=2.1,
        help="Multiplier for camera distance relative to object span",
    )
    parser.add_argument(
        "--camera-azimuth",
        type=float,
        default=35.0,
        help="Camera azimuth angle in degrees around the XZ plane",
    )
    parser.add_argument(
        "--camera-elevation",
        type=float,
        default=32.0,
        help="Camera elevation angle in degrees above the XZ plane",
    )
    parser.add_argument(
        "--transparent-background",
        action="store_true",
        help="Render with a transparent film background",
    )
    return parser.parse_args(argv)


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_obj(obj_path):
    before = {obj.name for obj in bpy.data.objects}
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=str(obj_path))
    else:
        bpy.ops.import_scene.obj(filepath=str(obj_path))

    imported = [obj for obj in bpy.data.objects if obj.name not in before and obj.type == "MESH"]
    if not imported:
        raise RuntimeError(f"No mesh objects were imported from {obj_path}")
    return imported


def join_mesh_objects(mesh_objects):
    if len(mesh_objects) == 1:
        return mesh_objects[0]

    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]
    bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def smooth_object(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_smooth()


def object_bounds(obj):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_corner = Vector(
        (
            min(corner.x for corner in corners),
            min(corner.y for corner in corners),
            min(corner.z for corner in corners),
        )
    )
    max_corner = Vector(
        (
            max(corner.x for corner in corners),
            max(corner.y for corner in corners),
            max(corner.z for corner in corners),
        )
    )
    center = (min_corner + max_corner) / 2.0
    size = max_corner - min_corner
    return min_corner, max_corner, center, size


def look_at(obj, target, track_axis="-Z", up_axis="Y"):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat(track_axis, up_axis).to_euler()


def create_material():
    material = bpy.data.materials.new(name="ProfilometrySurface")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = (0.78, 0.83, 0.9, 1.0)
        principled.inputs["Roughness"].default_value = 0.38
        principled.inputs["Metallic"].default_value = 0.08
        if "Specular IOR Level" in principled.inputs:
            principled.inputs["Specular IOR Level"].default_value = 0.55
        elif "Specular" in principled.inputs:
            principled.inputs["Specular"].default_value = 0.55
    return material


def assign_material(obj, material):
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)


def setup_world(transparent):
    world = bpy.data.worlds["World"]
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs[0].default_value = (0.025, 0.03, 0.04, 1.0)
        background.inputs[1].default_value = 0.85
    bpy.context.scene.render.film_transparent = transparent


def setup_camera(center, size, args):
    camera_data = bpy.data.cameras.new(name="RenderCamera")
    camera = bpy.data.objects.new("RenderCamera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera

    span = max(size.x, size.y, size.z, 1.0)
    radius = span * args.camera_distance_scale
    azimuth = math.radians(args.camera_azimuth)
    elevation = math.radians(args.camera_elevation)

    camera.location = Vector(
        (
            center.x + radius * math.cos(elevation) * math.cos(azimuth),
            center.y + radius * math.sin(elevation),
            center.z + radius * math.cos(elevation) * math.sin(azimuth),
        )
    )
    camera.data.lens = 55
    camera.data.clip_start = 0.001
    camera.data.clip_end = max(radius * 20.0, 100.0)
    look_at(camera, center)
    return camera


def setup_lights(center, size):
    span = max(size.x, size.z, 1.0)
    height = max(size.y, 0.1)

    key_data = bpy.data.lights.new(name="KeyLight", type="AREA")
    key_data.energy = 3500
    key_data.shape = "RECTANGLE"
    key_data.size = span * 1.5
    key = bpy.data.objects.new("KeyLight", key_data)
    key.location = Vector((center.x + span * 0.3, center.y + height * 3.0 + 1.0, center.z - span * 0.6))
    bpy.context.scene.collection.objects.link(key)
    look_at(key, center)

    fill_data = bpy.data.lights.new(name="FillLight", type="SUN")
    fill_data.energy = 1.8
    fill = bpy.data.objects.new("FillLight", fill_data)
    fill.location = Vector((center.x - span, center.y + height * 2.0 + 1.0, center.z + span))
    bpy.context.scene.collection.objects.link(fill)
    look_at(fill, center)


def output_format_for(path):
    suffix = path.suffix.lower()
    formats = {
        ".png": "PNG",
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".bmp": "BMP",
        ".tif": "TIFF",
        ".tiff": "TIFF",
        ".exr": "OPEN_EXR",
    }
    try:
        return formats[suffix]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported output extension '{suffix}'. Use one of: {', '.join(sorted(formats))}"
        ) from exc


def configure_render(output_path, args):
    scene = bpy.context.scene
    scene.render.engine = args.engine
    scene.render.resolution_x = args.resolution_x
    scene.render.resolution_y = args.resolution_y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = output_format_for(output_path)
    scene.render.filepath = str(output_path)

    if args.engine == "CYCLES":
        if not hasattr(scene, "cycles"):
            raise RuntimeError("Cycles is not available in this Blender build")
        scene.cycles.device = "CPU"
        scene.cycles.samples = args.samples
        if hasattr(scene.cycles, "use_denoising"):
            scene.cycles.use_denoising = True


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input OBJ not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Rendering OBJ: {input_path}")
    print(f"Output image:  {output_path}")

    reset_scene()
    imported = import_obj(input_path)
    obj = join_mesh_objects(imported)
    smooth_object(obj)

    material = create_material()
    assign_material(obj, material)

    _, _, center, size = object_bounds(obj)
    setup_world(args.transparent_background)
    setup_camera(center, size, args)
    setup_lights(center, size)
    configure_render(output_path, args)

    bpy.ops.render.render(write_still=True)
    print(f"Render saved to: {output_path}")


if __name__ == "__main__":
    main()
