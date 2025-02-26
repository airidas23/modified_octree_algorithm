# main.py
import os
import laspy
import numpy as np
import sys
from modified_octree import build_modified_octree
from visualization import visualize_octree_open3d, visualize_original

def assign_node_ids(root):
    from collections import deque
    queue = deque([root])
    node_id = 0
    while queue:
        node = queue.popleft()
        node.node_id = node_id
        node_id += 1
        for child in node.children:
            queue.append(child)
    return node_id  # total number of nodes


def main():
    try:
        # 1) Load LAS data
        las_path = "../data/2743_1234.las"
        
        # Check if file exists
        if not os.path.exists(las_path):
            raise FileNotFoundError(f"LAS file not found: '{las_path}'")
            
        print(f"Loading point cloud from {las_path}...")
        
        try:
            las_data = laspy.read(las_path)
        except Exception as e:
            raise ValueError(f"Failed to load LAS file: {str(e)}")
            
        # Extract coordinate data
        try:
            points = np.vstack([las_data.x, las_data.y, las_data.z]).T
        except Exception as e:
            raise ValueError(f"Invalid point data in LAS file: {str(e)}")
            
        # Check if we have any points
        if points.shape[0] == 0:
            raise ValueError("LAS file contains no points")
            
        print(f"Loaded {points.shape[0]} points from {las_path}")

        # 2) Build the Modified Octree
        max_depth = 6
        min_points = 100
        
        # Validate parameters
        if max_depth < 0:
            raise ValueError(f"max_depth must be non-negative: {max_depth}")
            
        if min_points < 1:
            raise ValueError(f"min_points must be at least 1: {min_points}")
            
        print(f"Building octree with max_depth={max_depth}, min_points={min_points}")
        try:
            root = build_modified_octree(points, max_depth=max_depth, min_points=min_points)
        except Exception as e:
            raise RuntimeError(f"Failed to build octree: {str(e)}")

        # 3) Assign unique node IDs
        try:
            total_nodes = assign_node_ids(root)
        except Exception as e:
            raise RuntimeError(f"Failed to assign node IDs: {str(e)}")
            
        print(f"Total nodes in octree: {total_nodes}")

        # 4) Visualize in Open3D
        print("Visualizing octree...")
        try:
            visualize_octree_open3d(root, points, total_nodes)
        except Exception as e:
            print(f"Visualization failed: {str(e)}")
            # Fallback to simple visualization if advanced viz fails
            print("Attempting to visualize original point cloud...")
            visualize_original(points)

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()


