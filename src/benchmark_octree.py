import tracemalloc
import time
import laspy
import numpy as np
import pandas as pd
from modified_octree import build_modified_octree

# Define where to save results
results_path = "../src/benchmark_results.csv"

# Load LAS file from user's data
las_path = "../data/2743_1234.las"

try:
    las_data = laspy.read(las_path)
    points = np.vstack([las_data.x, las_data.y, las_data.z]).T
except Exception as e:
    raise ValueError(f"Failed to load LAS file: {str(e)}")

# Parameters for Octree
max_depth = 6
min_points = 100

# Benchmark modified octree
tracemalloc.start()
start_time = time.time()
root = build_modified_octree(points, max_depth=max_depth, min_points=min_points)
modified_memory = tracemalloc.get_traced_memory()[1]  # Peak memory usage
modified_time = time.time() - start_time
tracemalloc.stop()


# Count nodes in modified octree
def count_nodes(node):
    queue = [node]
    node_count = 0
    while queue:
        node = queue.pop(0)
        node_count += 1
        queue.extend(node.children)
    return node_count


modified_nodes = count_nodes(root)


# Simulate a standard octree without spherical filtering for comparison
def build_standard_octree(points, max_depth):
    """ Simulated standard octree for comparison. """

    class StandardOctreeNode:
        def __init__(self, center, half_length, depth=0):
            self.center = np.array(center, dtype=float)
            self.half_length = float(half_length)
            self.depth = depth
            self.children = []

    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    center = 0.5 * (min_xyz + max_xyz)
    half_length = 0.5 * np.max(max_xyz - min_xyz)

    root = StandardOctreeNode(center, half_length, depth=0)
    queue = [(root, points, 0)]
    node_count = 0

    while queue:
        node, node_points, depth = queue.pop(0)
        node_count += 1

        if depth < max_depth and len(node_points) > min_points:
            child_half = node.half_length / 2.0
            offsets = np.array([
                [-1, -1, -1],
                [-1, -1, 1],
                [-1, 1, -1],
                [-1, 1, 1],
                [1, -1, -1],
                [1, -1, 1],
                [1, 1, -1],
                [1, 1, 1]
            ]) * child_half

            for offset in offsets:
                child_center = node.center + offset
                child_node = StandardOctreeNode(child_center, child_half, depth=depth + 1)
                queue.append((child_node, node_points, depth + 1))

    return node_count


# Benchmark standard octree
tracemalloc.start()
start_time = time.time()
standard_nodes = build_standard_octree(points, max_depth=max_depth)
standard_memory = tracemalloc.get_traced_memory()[1]  # Peak memory usage
standard_time = time.time() - start_time
tracemalloc.stop()

# Compute reductions
node_reduction = (standard_nodes - modified_nodes) / standard_nodes * 100
memory_reduction = (standard_memory - modified_memory) / standard_memory * 100

# Create results dataframe
df_results = pd.DataFrame([{
    "Standard Octree Nodes": standard_nodes,
    "Modified Octree Nodes": modified_nodes,
    "Node Reduction (%)": node_reduction,
    "Standard Octree Memory (MB)": standard_memory / 1e6,
    "Modified Octree Memory (MB)": modified_memory / 1e6,
    "Memory Reduction (%)": memory_reduction,
    "Standard Octree Time (s)": standard_time,
    "Modified Octree Time (s)": modified_time,
}])

# Save results to CSV
df_results.to_csv(results_path, index=False)
print(f"Benchmarking results saved to {results_path}")

# Display results
print(df_results)
