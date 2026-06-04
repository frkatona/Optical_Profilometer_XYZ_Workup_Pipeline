import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

MATERIAL_PRESETS = {
    "custom": None,
    "technical_clay": {
        "color": "#c7d3e5",
        "roughness": 0.38,
        "metallic": 0.08,
        "specular": 0.55,
    },
    "lab_ceramic": {
        "color": "#f4f7fb",
        "roughness": 0.22,
        "metallic": 0.0,
        "specular": 0.6,
    },
    "brushed_aluminum": {
        "color": "#c9d0db",
        "roughness": 0.24,
        "metallic": 0.72,
        "specular": 0.52,
    },
    "graphite_matte": {
        "color": "#556272",
        "roughness": 0.7,
        "metallic": 0.12,
        "specular": 0.35,
    },
}


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
        help="Camera azimuth angle in degrees around the XY plane",
    )
    parser.add_argument(
        "--camera-elevation",
        type=float,
        default=32.0,
        help="Camera elevation angle in degrees above the XY plane",
    )
    parser.add_argument(
        "--camera-lens",
        type=float,
        default=55.0,
        help="Camera lens length in millimeters",
    )
    parser.add_argument(
        "--rotation-x",
        type=float,
        default=0.0,
        help="Rotate the imported object around X in degrees",
    )
    parser.add_argument(
        "--rotation-y",
        type=float,
        default=0.0,
        help="Rotate the imported object around Y in degrees",
    )
    parser.add_argument(
        "--rotation-z",
        type=float,
        default=0.0,
        help="Rotate the imported object around Z in degrees",
    )
    parser.add_argument(
        "--height-scale",
        type=float,
        default=1.0,
        help="Scale the imported surface height along its local height axis",
    )
    parser.add_argument(
        "--auto-height-scale",
        action="store_true",
        help="Automatically scale relief to a target fraction of the lateral span",
    )
    parser.add_argument(
        "--auto-height-ratio",
        type=float,
        default=0.12,
        help="Target height-to-span ratio used when auto height scaling is enabled",
    )
    parser.add_argument(
        "--world-strength",
        type=float,
        default=0.85,
        help="Background light strength",
    )
    parser.add_argument(
        "--key-light-energy",
        type=float,
        default=3500.0,
        help="Key area light energy",
    )
    parser.add_argument(
        "--fill-light-energy",
        type=float,
        default=1.8,
        help="Fill sun light energy",
    )
    parser.add_argument(
        "--material-color",
        default="#c7d3e5",
        help="Base material color as a hex string, for example #c7d3e5",
    )
    parser.add_argument(
        "--material-preset",
        default="custom",
        choices=sorted(MATERIAL_PRESETS),
        help="Named material preset to use instead of the manual material values",
    )
    parser.add_argument(
        "--material-roughness",
        type=float,
        default=0.38,
        help="Material roughness value between 0 and 1",
    )
    parser.add_argument(
        "--material-metallic",
        type=float,
        default=0.08,
        help="Material metallic value between 0 and 1",
    )
    parser.add_argument(
        "--material-specular",
        type=float,
        default=0.55,
        help="Material specular or specular IOR level value between 0 and 1",
    )
    parser.add_argument(
        "--transparent-background",
        action="store_true",
        help="Render with a transparent film background",
    )
    parser.add_argument(
        "--shading",
        default="smooth",
        choices=["smooth", "flat"],
        help="Shading mode for the imported surface",
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


def apply_shading(obj, args):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    if args.shading == "smooth":
        bpy.ops.object.shade_smooth()
    else:
        bpy.ops.object.shade_flat()


def rotate_object(obj, args):
    import_rotation = obj.rotation_euler.copy()
    obj.rotation_euler = (
        import_rotation.x + math.radians(args.rotation_x),
        import_rotation.y + math.radians(args.rotation_y),
        import_rotation.z + math.radians(args.rotation_z),
    )


def apply_height_scale(obj, args):
    if args.height_scale <= 0:
        raise ValueError("Height scale must be greater than 0")
    scale_factor = args.height_scale

    if args.auto_height_scale:
        if args.auto_height_ratio <= 0:
            raise ValueError("Auto height ratio must be greater than 0")
        size = obj.dimensions.copy()
        lateral_span = max(size.x, size.z, 1e-6)
        height_span = size.y
        if height_span <= 1e-9:
            print("Auto height scaling skipped: imported surface has no measurable height span")
        else:
            auto_scale = (lateral_span * args.auto_height_ratio) / height_span
            scale_factor *= auto_scale
            print(
                "Auto height scaling enabled:"
                f" target ratio={args.auto_height_ratio:.3f},"
                f" computed factor={auto_scale:.3f}x"
            )

    obj.scale.y *= scale_factor
    if abs(scale_factor - 1.0) > 1e-9:
        print(f"Applied final height scale: {scale_factor:.3f}x")


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


def look_at(obj, target, track_axis="-Z", up_axis="Z"):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat(track_axis, up_axis).to_euler()


def hex_to_rgba(value):
    value = value.strip().lstrip("#")
    if len(value) != 6:
        raise ValueError("Material color must be a 6-digit hex string like #c7d3e5")
    r = int(value[0:2], 16) / 255.0
    g = int(value[2:4], 16) / 255.0
    b = int(value[4:6], 16) / 255.0
    return (r, g, b, 1.0)


def material_settings(args):
    preset = MATERIAL_PRESETS.get(args.material_preset)
    if preset is None:
        return {
            "color": args.material_color,
            "roughness": args.material_roughness,
            "metallic": args.material_metallic,
            "specular": args.material_specular,
        }
    return preset


def create_material(args):
    settings = material_settings(args)
    material = bpy.data.materials.new(name="ProfilometrySurface")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = hex_to_rgba(settings["color"])
        principled.inputs["Roughness"].default_value = settings["roughness"]
        principled.inputs["Metallic"].default_value = settings["metallic"]
        if "Specular IOR Level" in principled.inputs:
            principled.inputs["Specular IOR Level"].default_value = settings["specular"]
        elif "Specular" in principled.inputs:
            principled.inputs["Specular"].default_value = settings["specular"]
    return material


def assign_material(obj, material):
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)


def setup_world(args):
    scene = bpy.context.scene
    world = scene.world or bpy.data.worlds.get("World")
    if world is None:
        world = bpy.data.worlds.new(name="World")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    background = nodes.get("Background")
    if background is None:
        background = nodes.new(type="ShaderNodeBackground")

    output = nodes.get("World Output")
    if output is None:
        output = nodes.new(type="ShaderNodeOutputWorld")

    if not any(
        link.from_node == background and link.to_node == output for link in links
    ):
        links.new(background.outputs["Background"], output.inputs["Surface"])

    background.inputs[0].default_value = (0.025, 0.03, 0.04, 1.0)
    background.inputs[1].default_value = args.world_strength
    scene.render.film_transparent = args.transparent_background


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
            center.y + radius * math.cos(elevation) * math.sin(azimuth),
            center.z + radius * math.sin(elevation),
        )
    )
    camera.data.lens = args.camera_lens
    camera.data.clip_start = 0.001
    camera.data.clip_end = max(radius * 20.0, 100.0)
    look_at(camera, center)
    return camera


def setup_lights(center, size, args):
    span = max(size.x, size.y, 1.0)
    height = max(size.z, 0.1)

    key_data = bpy.data.lights.new(name="KeyLight", type="AREA")
    key_data.energy = args.key_light_energy
    key_data.shape = "RECTANGLE"
    key_data.size = span * 1.5
    key = bpy.data.objects.new("KeyLight", key_data)
    key.location = Vector(
        (
            center.x + span * 0.3,
            center.y - span * 0.6,
            center.z + height * 3.0 + 1.0,
        )
    )
    bpy.context.scene.collection.objects.link(key)
    look_at(key, center)

    fill_data = bpy.data.lights.new(name="FillLight", type="SUN")
    fill_data.energy = args.fill_light_energy
    fill = bpy.data.objects.new("FillLight", fill_data)
    fill.location = Vector(
        (
            center.x - span,
            center.y + span,
            center.z + height * 2.0 + 1.0,
        )
    )
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
            scene.cycles.use_denoising = bool(
                getattr(bpy.app.build_options, "openimagedenoise", False)
            )


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
    apply_shading(obj, args)
    apply_height_scale(obj, args)
    rotate_object(obj, args)

    material = create_material(args)
    assign_material(obj, material)

    _, _, center, size = object_bounds(obj)
    setup_world(args)
    setup_camera(center, size, args)
    setup_lights(center, size, args)
    configure_render(output_path, args)

    bpy.ops.render.render(write_still=True)
    print(f"Render saved to: {output_path}")


if __name__ == "__main__":
    main()
