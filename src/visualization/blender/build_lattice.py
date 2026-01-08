# visualization/blender/build_lattice.py

from __future__ import annotations

import math

import bpy
import numpy as np
from mathutils import Vector


def init_scene() -> None:
    """Clear the default Blender scene (objects, meshes, cameras, and lights)."""
    print("[build_lattice] init_scene(): clearing scene")
    print(
        "[build_lattice] Objects BEFORE clear:",
        [obj.name for obj in bpy.data.objects],
    )

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh, do_unlink=True)

    for cam in list(bpy.data.cameras):
        bpy.data.cameras.remove(cam, do_unlink=True)

    for lamp in list(bpy.data.lights):
        bpy.data.lights.remove(lamp, do_unlink=True)

    print(
        "[build_lattice] Objects AFTER clear:",
        [obj.name for obj in bpy.data.objects],
    )


def build_cloth_object(
    positions: np.ndarray,
    triangles: np.ndarray,
    name: str = "Cloth",
) -> bpy.types.Object:
    """Create a Blender mesh object from the first frame of the simulation."""
    print(f"[build_lattice] positions.shape = {positions.shape}")
    print(f"[build_lattice] triangles.shape = {triangles.shape}")

    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError(
            "[build_lattice] Expected positions shape (T, N, 3), "
            f"got {positions.shape}"
        )
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(
            "[build_lattice] Expected triangles shape (K, 3), "
            f"got {triangles.shape}"
        )

    _, n_verts, _ = positions.shape
    n_faces, _ = triangles.shape

    max_index = int(triangles.max())
    print(
        f"[build_lattice] N = {n_verts}, max(triangles) = {max_index}, K = {n_faces}"
    )
    if max_index >= n_verts:
        raise ValueError(
            f"[build_lattice] Triangle index {max_index} out of range for N={n_verts}"
        )

    verts0 = positions[0]
    verts_list = [tuple(v) for v in verts0]
    faces_list = [tuple(f) for f in triangles]

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts_list, [], faces_list)
    mesh.update(calc_edges=True)

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    print(
        f"[build_lattice] Created cloth object '{name}' with {n_verts} verts and "
        f"{n_faces} faces."
    )
    print("[build_lattice] Objects NOW:", [o.name for o in bpy.data.objects])

    return obj


def create_camera_and_light_for_object(
    obj: bpy.types.Object,
    distance_factor: float = 3.0,
) -> None:
    """Create a camera and a light relative to the cloth object."""
    scene = bpy.context.scene

    bbox_world = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    center = sum(bbox_world, Vector((0.0, 0.0, 0.0))) / 8.0

    xs = [v.x for v in bbox_world]
    ys = [v.y for v in bbox_world]
    zs = [v.z for v in bbox_world]
    size_x = max(xs) - min(xs)
    size_y = max(ys) - min(ys)
    size_z = max(zs) - min(zs)
    max_size = max(size_x, size_y, size_z, 1e-3)

    dist = distance_factor * max_size
    view_dir = Vector((0.0, -1.0, 0.0))
    cam_location = center - dist * view_dir

    cam_data = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)

    cam_obj.location = cam_location
    cam_obj.rotation_euler = view_dir.to_track_quat("-Z", "Y").to_euler()

    scene.camera = cam_obj
    print(
        "[build_lattice] Created camera at "
        f"{cam_obj.location}, looking along -y at {center}"
    )

    light_data = bpy.data.lights.new(name="Sun", type="SUN")
    light_obj = bpy.data.objects.new(name="Sun", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)

    light_obj.location = center + Vector((dist, dist, dist))
    light_data.energy = 3.0
    light_data.angle = math.radians(15.0)

    print(f"[build_lattice] Created sun light at {light_obj.location}")

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs["Color"].default_value = (0.02, 0.02, 0.02, 1.0)
        bg.inputs["Strength"].default_value = 0.5


def create_floor_for_object(
    obj: bpy.types.Object,
    size_factor: float = 5.0,
    offset_factor: float = 0.2,
) -> bpy.types.Object:
    """Create a floor plane under the given object for visual reference."""
    bbox_world = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    xs = [v.x for v in bbox_world]
    ys = [v.y for v in bbox_world]
    zs = [v.z for v in bbox_world]

    center = sum(bbox_world, Vector((0.0, 0.0, 0.0))) / 8.0
    size_x = max(xs) - min(xs)
    size_y = max(ys) - min(ys)
    size_z = max(zs) - min(zs)
    max_size = max(size_x, size_y, size_z, 1e-3)

    floor_size = size_factor * max_size
    floor_z = min(zs) - offset_factor * max_size

    bpy.ops.mesh.primitive_plane_add(
        size=1.0,
        enter_editmode=False,
        align="WORLD",
        location=(center.x, center.y, floor_z),
    )
    floor_obj = bpy.context.active_object
    floor_obj.name = "Floor"
    floor_obj.scale = (floor_size, floor_size, 1.0)

    mat = bpy.data.materials.new(name="FloorMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.8
    floor_obj.data.materials.append(mat)

    print(f"[build_lattice] Created floor at z={floor_z}, size={floor_size}")
    return floor_obj
