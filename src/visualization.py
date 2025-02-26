import open3d as o3d
import numpy as np
import colorsys
from collections import deque

def visualize_octree_open3d(root, points, total_nodes, minimized_mode=False):
    """
    Visualizes the modified Octree in an Open3D window:
     - Nodes show a wireframe cube and a sphere.
     - Points are colored by node ID.
     - Displays a coordinate system.
    
    Parameters:
        root: Root node (with node_id)
        points: (N, 3) array of point coordinates.
        total_nodes: Total number of nodes for color normalization.
        minimized_mode: If True, captures a screenshot instead of running the GUI loop.
    """
    geometry_list = []
    spheres_list = []  # List of sphere meshes with colors
    queue = deque([root])
    
    all_points = []
    all_colors = []
    
    while queue:
        node = queue.popleft()
        node_color = _get_node_color(node.node_id, total_nodes)
        
        # Draw wireframe cube
        line_set = _create_bounding_box_lines(node.center, node.half_length, node_color)
        geometry_list.append(line_set)
        
        # Draw sphere (stored separately for transparency)
        sphere = _create_sphere(node.center, node.sphere_radius, node_color)
        spheres_list.append((sphere, node_color))
        
        # Collect node points with color
        node_points = points[node.point_indices]
        if len(node_points) > 0:
            all_points.append(node_points)
            color_array = np.tile(node_color, (len(node_points), 1))
            all_colors.append(color_array)
        
        for child in node.children:
            queue.append(child)
    
    # Merge all node points into a single point cloud
    if len(all_points) > 0:
        all_points = np.vstack(all_points)
        all_colors = np.vstack(all_colors)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors)
        if len(all_points) > 16777213:
            print("Downsampling point cloud for selection limits.")
            voxel_size = np.linalg.norm(all_points.max(axis=0) - all_points.min(axis=0)) / 100.0
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        geometry_list.append(pcd)
    
    # Initialize Open3D GUI app
    o3d.visualization.gui.Application.instance.initialize()
    
    # Create visualizer with transparency support
    vis = o3d.visualization.O3DVisualizer("Octree Visualization", 1024, 768)
    
    # Add non-sphere geometries
    for idx, geom in enumerate(geometry_list):
        vis.add_geometry(f"geom_{idx}", geom, None)
    
    # Add spheres with alpha transparency (0.5)
    for idx, (sphere, color) in enumerate(spheres_list):
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"
        mat.base_color = [color[0], color[1], color[2], 0.5]
        mat.base_metallic = 0.0
        mat.base_roughness = 0.5
        vis.add_geometry(f"sphere_{idx}", sphere, mat)
    
    # Set up camera based on point cloud bounds
    if len(all_points) > 0:
        pts = np.vstack(all_points)
        min_bound = pts.min(axis=0)
        max_bound = pts.max(axis=0)
        center = (min_bound + max_bound) / 2.0
        diag = np.linalg.norm(max_bound - min_bound)
        if diag < 1e-6:
            diag = 1.0
        eye = [center[0], center[1], center[2] + diag * 2.0]
        vis.setup_camera(60, center, eye, [0, 1, 0])
    
    if minimized_mode:
        vis.poll_events()
        vis.update_renderer()
        filename = "temp_vis.png"
        vis.capture_screen_image(filename)
        o3d.visualization.gui.Application.instance.destroy()
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        img = mpimg.imread(filename)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    else:
        o3d.visualization.gui.Application.instance.run()
        o3d.visualization.gui.Application.instance.destroy()

def _get_node_color(node_id, total_nodes):
    """
    Converts node_id to a unique RGB color using HSV.
    """
    if total_nodes > 1:
        h = float(node_id) / float(total_nodes - 1)
    else:
        h = 0.5
    r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
    return (r, g, b)

def _create_bounding_box_lines(center, half_length, color):
    """
    Creates a LineSet outlining a cube.
    """
    c = center
    h = half_length
    corners = np.array([
        [c[0]-h, c[1]-h, c[2]-h],
        [c[0]-h, c[1]-h, c[2]+h],
        [c[0]-h, c[1]+h, c[2]-h],
        [c[0]-h, c[1]+h, c[2]+h],
        [c[0]+h, c[1]-h, c[2]-h],
        [c[0]+h, c[1]-h, c[2]+h],
        [c[0]+h, c[1]+h, c[2]-h],
        [c[0]+h, c[1]+h, c[2]+h],
    ])
    lines = [
        [0, 1], [0, 2], [0, 4],
        [1, 3], [1, 5],
        [2, 3], [2, 6],
        [3, 7],
        [4, 5], [4, 6],
        [5, 7],
        [6, 7]
    ]
    colors = [color for _ in lines]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def _create_sphere(center, radius, color):
    """
    Creates a sphere mesh at the given center and radius.
    """
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(center)
    sphere.compute_vertex_normals()  # Compute normals for shading
    return sphere

def visualize_original(points, fast_mode=True):
    """
    Visualizes the original point cloud without filters.
    
    Parameters:
        points: (N, 3) NumPy array with point coordinates.
        fast_mode: If True, downsamples for faster rendering.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if fast_mode and len(points) > 100000:
        pts = np.asarray(points)
        bbox = pts.ptp(axis=0)
        voxel_size = np.linalg.norm(bbox) / 200.0
        print(f"Downsampling point cloud with voxel size: {voxel_size}")
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    o3d.visualization.draw_geometries([pcd], window_name="Original Visualization", width=1024, height=768)