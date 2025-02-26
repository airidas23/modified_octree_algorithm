import numpy as np
import laspy
from sklearn.preprocessing import RobustScaler

def process_las_file(file_path):
    """
    Reads a LAS file and processes the point cloud, filtering points outside the root sphere.
    
    Args:
        file_path (str): Path to the LAS file
        
    Returns:
        points_norm (numpy.ndarray): Normalized (N, 3) array of point coordinates
        intensities (numpy.ndarray): Array of intensity values
    """
    # Read the LAS file
    las = laspy.read(file_path)
    
    # Extract points and intensity
    points = np.vstack((las.x, las.y, las.z)).T
    intensities = np.asarray(las.intensity)
    
    # Normalize points
    scaler = RobustScaler()
    points_norm = scaler.fit_transform(points)
    
    # Filter points outside the root sphere
    # Find bounding box center and radius
    min_xyz = points_norm.min(axis=0)
    max_xyz = points_norm.max(axis=0)
    center = 0.5 * (min_xyz + max_xyz)
    radius = np.linalg.norm(max_xyz - center)
    
    # Keep only points inside the sphere
    dist_sq = np.sum((points_norm - center)**2, axis=1)
    in_sphere_mask = dist_sq <= (radius**2)
    
    return points_norm[in_sphere_mask], intensities[in_sphere_mask]
