from src.utils.typing_utils import *

import os
import numpy as np
from PIL import Image
import trimesh
from trimesh.transformations import rotation_matrix
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    TexturesVertex,
    TexturesUV
)
from diffusers.utils import export_to_video
from diffusers.utils.loading_utils import load_video
import torch
from torchvision.utils import make_grid

def _pytorch3d_render(
    mesh: trimesh.Trimesh,
    camera_pose: np.ndarray,
    image_size=(512, 512),
    fov=40.0,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    # 1. PREPARE MESH DATA
    verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.int64, device=device)

    # 2. HANDLE TEXTURES (Robust to PBR vs Simple Materials)
    texture_image = None
    
    # Check if visual is texture-based
    if hasattr(mesh.visual, 'kind') and mesh.visual.kind == 'texture':
        material = mesh.visual.material
        # Case A: Standard SimpleMaterial (OBJ, etc.)
        if hasattr(material, 'image') and material.image is not None:
            texture_image = material.image
        # Case B: PBRMaterial (GLB, GLTF) - Texture is in baseColorTexture
        elif hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
            texture_image = material.baseColorTexture

    if texture_image is not None:
        # Ensure image is RGB
        if texture_image.mode != 'RGB':
            texture_image = texture_image.convert('RGB')
        
        # Convert to tensor [1, H, W, 3] and normalize to [0, 1]
        image_np = np.array(texture_image).astype(np.float32) / 255.0
        maps = torch.tensor(image_np, device=device).unsqueeze(0) # [1, H, W, 3]

        # Prepare UV coordinates
        # Trimesh stores UVs in mesh.visual.uv matching vertex count
        verts_uvs = torch.tensor(mesh.visual.uv, dtype=torch.float32, device=device).unsqueeze(0) # [1, V, 2]
        
        # In Trimesh, UV indices usually match vertex indices (dense UVs)
        faces_uvs = faces.unsqueeze(0) # [1, F, 3]

        textures = TexturesUV(maps=maps, faces_uvs=faces_uvs, verts_uvs=verts_uvs)

    # Fallback to Vertex Colors if available
    elif mesh.visual.kind == "vertex" and mesh.visual.vertex_colors is not None:
        colors = mesh.visual.vertex_colors[:, :3] / 255.0
        textures = TexturesVertex(
            verts_features=torch.tensor(colors, dtype=torch.float32, device=device)[None]
        )
    # Fallback to plain Grey
    else:
        textures = TexturesVertex(
            verts_features=torch.ones_like(verts)[None] * 0.7
        )

    mesh_p3d = Meshes(
        verts=[verts],
        faces=[faces],
        textures=textures
    )

    # 3. CAMERA TRANSFORMATION
    # Calculate World-to-View matrix (Inverse of Camera Pose)
    w2v = np.linalg.inv(camera_pose)
    
    # Convert to Tensor
    R_tensor = torch.tensor(w2v[:3, :3], dtype=torch.float32, device=device)
    T_tensor = torch.tensor(w2v[:3, 3], dtype=torch.float32, device=device)
    
    # Coordinate Flip: OpenGL (-Z render) -> PyTorch3D (+Z render)
    # Flip X and Z axes
    flip_mat = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=torch.float32, device=device)
    
    R_p3d = flip_mat @ R_tensor
    T_p3d = flip_mat @ T_tensor

    # Format for PyTorch3D cameras
    R_p3d = R_p3d.t().unsqueeze(0)
    T_p3d = T_p3d.unsqueeze(0)

    cameras = FoVPerspectiveCameras(
        R=R_p3d,
        T=T_p3d,
        fov=fov,
        device=device
    )

    # 4. LIGHTING
    # "Headlight" setup using original camera position in world space
    light_location = torch.tensor(camera_pose[:3, 3], dtype=torch.float32, device=device).unsqueeze(0)
    
    lights = PointLights(
        location=light_location,
        diffuse_color=[[0.8, 0.8, 0.8]],
        specular_color=[[0.2, 0.2, 0.2]],
        device=device
    )

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            cameras=cameras,
            lights=lights,
            device=device
        )
    )

    images = renderer(mesh_p3d)  # [1, H, W, 4]
    image = images[0, ..., :3].clamp(0, 1).cpu().numpy()
    image = (image * 255).astype(np.uint8)

    return image

def rotation_matrix_from_vectors(vec1, vec2):
    a, b = vec1 / np.linalg.norm(vec1), vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s == 0:
        return np.eye(3) if c > 0 else -np.eye(3)
    kmat = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))

def create_circular_camera_positions(
    num_views: int,
    radius: float,
    axis: np.ndarray = np.array([0.0, 1.0, 0.0])
) -> List[np.ndarray]:
    positions = []
    axis = axis / np.linalg.norm(axis)
    for i in range(num_views):
        theta = 2 * np.pi * i / num_views
        position = np.array([
            np.sin(theta) * radius,
            0.0,
            np.cos(theta) * radius
        ])
        if not np.allclose(axis, np.array([0.0, 1.0, 0.0])):
            R = rotation_matrix_from_vectors(np.array([0.0, 1.0, 0.0]), axis)
            position = R @ position
        positions.append(position)
    return positions

def create_circular_camera_poses(
    num_views: int,
    radius: float,
    axis: np.ndarray = np.array([0.0, 1.0, 0.0])
) -> List[np.ndarray]:
    canonical_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, radius],
        [0.0, 0.0, 0.0, 1.0]
    ])
    poses = []
    for i in range(num_views):
        theta = 2 * np.pi * i / num_views
        R = rotation_matrix(
            angle=theta,
            direction=axis,
            point=[0, 0, 0]
        )
        pose = R @ canonical_pose
        poses.append(pose)
    return poses

def render_views_around_mesh(
    mesh: Union[trimesh.Trimesh, trimesh.Scene],
    num_views: int = 36,
    radius: float = 3.5,
    axis: np.ndarray = np.array([0.0, 1.0, 0.0]),
    image_size: tuple = (512, 512),
    fov: float = 40.0,
    light_intensity: Optional[float] = 5.0,
    znear: float = 0.1,
    zfar: float = 10.0, 
    normalize_depth: bool = False,
    flags: int = 0,
    return_depth: bool = False,
    return_type: Literal['pil', 'ndarray'] = 'pil'
):

    if not isinstance(mesh, (trimesh.Trimesh, trimesh.Scene)):
        raise ValueError("mesh must be a trimesh.Trimesh or trimesh.Scene object")

    images = []
    for i in range(num_views):
        azimuth = 360.0 * i / num_views
        img = render_single_view(
            mesh,
            azimuth=azimuth,
            elevation=0.0,
            radius=radius,
            image_size=image_size,
            fov=fov,
            return_type=return_type
        )
        images.append(img)

    return images

def render_normal_views_around_mesh(
    mesh: Union[trimesh.Trimesh, trimesh.Scene],
    num_views: int = 36,
    radius: float = 3.5,
    axis: np.ndarray = np.array([0.0, 1.0, 0.0]),
    image_size: tuple = (512, 512),
    fov: float = 40.0,
    light_intensity: Optional[float] = 5.0,
    znear: float = 0.1,
    zfar: float = 10.0,
    normalize_depth: bool = False,
    flags: int = 0,
    return_depth: bool = False, 
    return_type: Literal['pil', 'ndarray'] = 'pil'
) -> Union[
        List[Image.Image], 
        List[np.ndarray], 
        Tuple[List[Image.Image], List[Image.Image]], 
        Tuple[List[np.ndarray], List[np.ndarray]]
    ]:
    
    if not isinstance(mesh, (trimesh.Trimesh, trimesh.Scene)):
        raise ValueError("mesh must be a trimesh.Trimesh or trimesh.Scene object")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    normals = mesh.vertex_normals
    colors = ((normals + 1.0) / 2.0 * 255).astype(np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        vertex_colors=colors
    )
    mesh = trimesh.Scene(mesh)
    return render_views_around_mesh(
        mesh, num_views, radius, axis, 
        image_size, fov, light_intensity, znear, zfar, 
        normalize_depth, 
        return_depth, return_type
    )

def create_camera_pose_on_sphere(
    azimuth: float = 0.0, 
    elevation: float = 0.0,
    radius: float = 3.5,
) -> np.ndarray:
    canonical_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, radius],
        [0.0, 0.0, 0.0, 1.0]
    ])
    azimuth = np.deg2rad(azimuth)
    elevation = np.deg2rad(elevation)
    position = np.array([
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
        np.cos(elevation) * np.cos(azimuth),
    ])
    R = np.eye(4)
    R[:3, :3] = rotation_matrix_from_vectors(
        np.array([0.0, 0.0, 1.0]), 
        position
    )
    pose = R @ canonical_pose
    return pose

def render_single_view(
    mesh: Union[trimesh.Trimesh, trimesh.Scene],
    azimuth: float = 0.0,
    elevation: float = 0.0,
    radius: float = 3.5,
    image_size: tuple = (512, 512),
    fov: float = 40.0,
    light_intensity: Optional[float] = 5.0,
    num_env_lights: int = 0,
    znear: float = 0.1,
    zfar: float = 10.0,
    normalize_depth: bool = False,
    flags: int = 0,
    return_depth: bool = False,
    return_type: Literal['pil', 'ndarray'] = 'pil'
):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()

    camera_pose = create_camera_pose_on_sphere(
        azimuth, elevation, radius
    )

    img = _pytorch3d_render(
        mesh,
        camera_pose,
        image_size=image_size,
        fov=fov
    )

    if return_type == "pil":
        img = Image.fromarray(img)

    return img

def render_normal_single_view(
    mesh: Union[trimesh.Trimesh, trimesh.Scene],
    azimuth: float = 0.0, 
    elevation: float = 0.0, 
    radius: float = 3.5,
    image_size: tuple = (512, 512),
    fov: float = 40.0,
    light_intensity: Optional[float] = 5.0,
    znear: float = 0.1,
    zfar: float = 10.0,
    normalize_depth: bool = False,
    return_depth: bool = False,
    return_type: Literal['pil', 'ndarray'] = 'pil'
) -> Union[
        Image.Image,
        np.ndarray,
        Tuple[Image.Image, Image.Image],
        Tuple[np.ndarray, np.ndarray]
    ]:
    if not isinstance(mesh, (trimesh.Trimesh, trimesh.Scene)):
        raise ValueError("mesh must be a trimesh.Trimesh or trimesh.Scene object")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    normals = mesh.vertex_normals
    colors = ((normals + 1.0) / 2.0 * 255).astype(np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        vertex_colors=colors
    )
    mesh = trimesh.Scene(mesh)
    return render_single_view(
        mesh, azimuth, elevation, radius, 
        image_size, fov, light_intensity, znear, zfar,
        normalize_depth, 0, 
        return_depth, return_type
    )

def export_renderings(
    images: List[Image.Image],
    export_path: str,
    fps: int = 36, 
    loop: int = 0
): 
    export_type = export_path.split('.')[-1]
    if export_type == 'mp4':
        export_to_video(
            images,
            export_path,
            fps=fps,
        )
    elif export_type == 'gif':
        duration = 1000 / fps
        images[0].save(
            export_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop
        )
    else:
        raise ValueError(f'Unknown export type: {export_type}')
    
def make_grid_for_images_or_videos(
    images_or_videos: Union[List[Image.Image], List[List[Image.Image]]],
    nrow: int = 4, 
    padding: int = 0, 
    pad_value: int = 0, 
    image_size: tuple = (512, 512),
    return_type: Literal['pil', 'ndarray'] = 'pil'
) -> Union[Image.Image, List[Image.Image], np.ndarray]:
    if isinstance(images_or_videos[0], Image.Image):
        images = [np.array(image.resize(image_size).convert('RGB')) for image in images_or_videos]
        images = np.stack(images, axis=0).transpose(0, 3, 1, 2) 
        images = torch.from_numpy(images)
        image_grid = make_grid(
            images,
            nrow=nrow,
            padding=padding,
            pad_value=pad_value,
            normalize=False
        ) 
        image_grid = image_grid.cpu().numpy()
        if return_type == 'pil':
            image_grid = Image.fromarray(image_grid.transpose(1, 2, 0))
        return image_grid
    elif isinstance(images_or_videos[0], list) and isinstance(images_or_videos[0][0], Image.Image):
        image_grids = []
        for i in range(len(images_or_videos[0])):
            images = [video[i] for video in images_or_videos]
            image_grid = make_grid_for_images_or_videos(
                images,
                nrow=nrow,
                padding=padding,
                return_type=return_type
            )
            image_grids.append(image_grid)
        if return_type == 'ndarray':
            image_grids = np.stack(image_grids, axis=0)
        return image_grids
    else:
        raise ValueError(f'Unknown input type: {type(images_or_videos[0])}')


def save_mesh_and_renderings(
    mesh: trimesh.Trimesh,
    save_dir: str,
    mesh_filename: str = "object.glb",
    rendering_prefix: str = "rendering",
    render_cfg: Dict = None,
    input_image_pil: Optional[Image.Image] = None,
    fps: int = 18,
    num_views: int = 36,
    radius: float = 4.0
):
    """
    Save a mesh and its renderings (images, normals, grids) to a directory.

    Args:
        mesh: The mesh to save and render.
        save_dir: Directory to save files.
        mesh_filename: Filename for the mesh (e.g., "object.glb").
        rendering_prefix: Prefix for rendering files (e.g., "rendering").
        render_cfg: Rendering config dict, if None, use defaults.
        input_image_pil: Input image for grid, optional.
        fps: FPS for GIFs.
        num_views: Number of views for rendering.
        radius: Camera radius.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    mesh.export(os.path.join(save_dir, mesh_filename))

    if render_cfg is None:
        render_cfg = {}
    num_views = render_cfg.get('num_views', num_views)
    radius = render_cfg.get('radius', radius)
    fps = render_cfg.get('fps', fps)

    rendered_images = render_views_around_mesh(mesh, num_views=num_views, radius=radius)
    rendered_normals = render_normal_views_around_mesh(mesh, num_views=num_views, radius=radius)
    
    grids = [rendered_images, rendered_normals]
    if input_image_pil is not None:
        grids.insert(0, [input_image_pil] * len(rendered_images))
    
    rendered_grids = make_grid_for_images_or_videos(grids, nrow=3 if input_image_pil else 2)
    
    export_renderings(rendered_images, os.path.join(save_dir, f"{rendering_prefix}.gif"), fps=fps)
    export_renderings(rendered_normals, os.path.join(save_dir, f"{rendering_prefix}_normal.gif"), fps=fps)
    export_renderings(rendered_grids, os.path.join(save_dir, f"{rendering_prefix}_grid.gif"), fps=fps)
    
    rendered_images[0].save(os.path.join(save_dir, f"{rendering_prefix}.png"))
    rendered_normals[0].save(os.path.join(save_dir, f"{rendering_prefix}_normal.png"))
    rendered_grids[0].save(os.path.join(save_dir, f"{rendering_prefix}_grid.png"))