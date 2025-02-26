"""
Implementation of a Modified Octree Algorithm with Embedded Spherical Subdivision Regions

This module defines a BFS-based modified octree construction:
- Each node is a bounding cube with a sphere at its center.
- Points outside the sphere are discarded.
- Points inside the sphere are recursively subdivided into 8 children.
"""

# modified_octree.py
import numpy as np

class ModifiedOctreeNode:
    def __init__(self, center, half_length, depth=0):
        self.center = np.array(center, dtype=float)
        self.half_length = float(half_length)
        self.depth = depth

        # Sphere radius = half the cube's edge length
        self.sphere_radius = self.half_length

        # Points that lie in this node's sphere
        self.point_indices = []

        # Child nodes
        self.children = []

def build_modified_octree(points, max_depth=5, min_points=20):
    """
    Build a modified octree from input points.
    
    Args:
        points (numpy.ndarray): Array of shape (N, 3) containing point coordinates
        max_depth (int): Maximum depth of the octree
        min_points (int): Minimum number of points required to subdivide a node
    
    Returns:
        ModifiedOctreeNode: Root node of the octree
        
    Raises:
        ValueError: If points array is invalid or parameters are incorrect
    """
    # Validate inputs
    if not isinstance(points, np.ndarray):
        raise ValueError("Points must be a numpy ndarray")
        
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Points must have shape (N, 3), got {points.shape}")
        
    if len(points) == 0:
        raise ValueError("Empty point cloud, no points to process")
        
    if max_depth < 0:
        raise ValueError(f"max_depth must be non-negative, got {max_depth}")
        
    if min_points < 1:
        raise ValueError(f"min_points must be at least 1, got {min_points}")
    
    # Check for NaN or infinite values
    if np.any(~np.isfinite(points)):
        raise ValueError("Point cloud contains NaN or infinite values")
    
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    
    # Check if point cloud is degenerate (all points on a line or plane)
    extent = max_xyz - min_xyz
    if np.any(extent <= 0):
        raise ValueError(f"Degenerate point cloud detected: extent={extent}")
        
    center = 0.5 * (min_xyz + max_xyz)
    half_length = 0.5 * np.max(extent)

    root = ModifiedOctreeNode(center, half_length, depth=0)
    root.point_indices = np.arange(len(points))

    from collections import deque
    queue = deque([root])

    while queue:
        node = queue.popleft()
        node_points = points[node.point_indices]

        # Keep only points within this node's sphere
        dist_sq = np.sum((node_points - node.center)**2, axis=1)
        in_sphere_mask = dist_sq <= (node.sphere_radius**2)
        sphere_indices = node.point_indices[in_sphere_mask]
        node.point_indices = sphere_indices

        # Subdivide if conditions are met
        if (node.depth < max_depth) and (len(sphere_indices) >= min_points):
            child_half = node.half_length / 2.0
            offsets = np.array([
                [-1, -1, -1],
                [-1, -1,  1],
                [-1,  1, -1],
                [-1,  1,  1],
                [ 1, -1, -1],
                [ 1, -1,  1],
                [ 1,  1, -1],
                [ 1,  1,  1]
            ]) * child_half

            parent_points = points[node.point_indices]
            for offset in offsets:
                child_center = node.center + offset
                child_node = ModifiedOctreeNode(child_center, child_half, depth=node.depth + 1)

                # Check which points from parent are in child's sphere
                dist_sq_child = np.sum((parent_points - child_center)**2, axis=1)
                in_child_sphere = dist_sq_child <= (child_node.sphere_radius**2)

                child_indices = node.point_indices[in_child_sphere]
                if len(child_indices) == 0:
                    continue

                child_node.point_indices = child_indices
                node.children.append(child_node)
                queue.append(child_node)

    return root
