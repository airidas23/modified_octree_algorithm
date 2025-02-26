import unittest
import numpy as np
import sys
import os

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.modified_octree import build_modified_octree

class TestModifiedOctree(unittest.TestCase):

    def test_points_inside_spheres(self):
        """Test that after each subdivision, all points in a node are inside its sphere."""
        # Create a random point cloud
        np.random.seed(42)  # For reproducibility
        points = np.random.uniform(-1, 1, size=(100, 3))
        
        # Build the octree
        max_depth = 4
        min_points = 100
        root = build_modified_octree(points, max_depth, min_points)
        
        # Check all nodes to ensure their points are inside their spheres
        def verify_points_in_sphere(node, all_points):
            node_points = all_points[node.point_indices]
            
            # Calculate distances from points to node center
            distances_squared = np.sum((node_points - node.center)**2, axis=1)
            
            # Check if all points are inside the sphere (distance <= radius)
            points_inside = np.all(distances_squared <= (node.sphere_radius**2))
            
            self.assertTrue(points_inside, 
                        f"Found points outside the sphere in node at depth {node.depth}")
            
            # Recurse for all children
            for child in node.children:
                verify_points_in_sphere(child, all_points)
        
            # Start verification from root
            verify_points_in_sphere(root, points)
            
            # Also verify that a point outside all spheres is properly rejected
            outside_point = np.array([[10.0, 10.0, 10.0]])  # Far outside any node
            all_points_with_outside = np.vstack([points, outside_point])
            
            # Build a new octree with the outside point included
            new_root = build_modified_octree(all_points_with_outside, max_depth, min_points)
            
            # Collect all point indices from the entire tree
            all_indices = set()
            
            def collect_indices(node):
                all_indices.update(node.point_indices)
                for child in node.children:
                    collect_indices(child)
            
            collect_indices(new_root)
            
            # The outside point should be the last one (index = len(points))
            self.assertNotIn(len(points), all_indices, 
                        "Outside point was incorrectly included in the octree")

if __name__ == '__main__':
    unittest.main()
