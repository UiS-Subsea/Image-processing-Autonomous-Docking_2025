import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

def visualize_point_cloud(ply_path, method='open3d'):
    """
    Visualize a point cloud from a PLY file using either Open3D or Matplotlib
    
    Parameters:
    -----------
    ply_path : str
        Path to the .ply file
    method : str
        'open3d' or 'matplotlib'
    """
    if method.lower() == 'open3d':
        # Open3D visualization
        pcd = o3d.io.read_point_cloud(ply_path)
        o3d.visualization.draw_geometries([pcd])
        
    elif method.lower() == 'matplotlib':
        # Read PLY file manually for Matplotlib
        vertices = []
        colors = []
        with open(ply_path, 'r') as f:
            # Skip header
            line = f.readline().strip()
            while line != 'end_header':
                line = f.readline().strip()
            
            # Read vertices and colors
            for line in f:
                if line.strip():
                    values = line.strip().split()
                    vertices.append([float(x) for x in values[:3]])
                    colors.append([int(x)/255.0 for x in values[3:]])
        
        vertices = np.array(vertices)
        colors = np.array(colors)
        
        # Create 3D scatter plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points
        scatter = ax.scatter(vertices[:, 0], 
                           vertices[:, 1], 
                           vertices[:, 2],
                           c=colors,
                           s=1)
        
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Auto-scale axes
        max_range = np.array([vertices[:, 0].max()-vertices[:, 0].min(),
                            vertices[:, 1].max()-vertices[:, 1].min(),
                            vertices[:, 2].max()-vertices[:, 2].min()]).max() / 2.0
        
        mid_x = (vertices[:, 0].max()+vertices[:, 0].min()) * 0.5
        mid_y = (vertices[:, 1].max()+vertices[:, 1].min()) * 0.5
        mid_z = (vertices[:, 2].max()+vertices[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.show()
    else:
        raise ValueError("Method must be either 'open3d' or 'matplotlib'")


if __name__ == "__main__":
    ply_file = "output/reconstruction.ply"
    visualize_point_cloud(ply_file, method='open3d')
    
    # Or use Matplotlib visualization
    #visualize_point_cloud(ply_file, method='matplotlib')