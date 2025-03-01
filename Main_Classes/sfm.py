import cv2
import numpy as np
import os
import time
from collections import defaultdict
from scipy.optimize import least_squares
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

class CameraParameters:
    """Manages camera parameters and operations"""
    
    def __init__(self, width, height, scale_factor=1.0):
        # Estimate focal length as image width
        self.f_x = width / scale_factor
        self.f_y = width / scale_factor  # Square pixels assumption
        self.c_x = width / (2 * scale_factor)
        self.c_y = height / (2 * scale_factor)
        self.width = width // scale_factor
        self.height = height // scale_factor
        
        # Build intrinsic matrix
        self.K = np.array([
            [self.f_x, 0, self.c_x],
            [0, self.f_y, self.c_y],
            [0, 0, 1]
        ], dtype=np.float32)
    
    def get_projection_matrix(self, R, t):
        """Constructs the camera projection matrix P = K[R|t]"""
        Rt = np.hstack((R, t.reshape(3, 1)))
        return np.matmul(self.K, Rt)


class ImageCollection:
    """Manages loading and preprocessing of images"""
    
    def __init__(self, image_dir, scale_factor=2.0):
        self.scale_factor = scale_factor
        self.images = []
        self.image_paths = []
        self.camera = None
        
        # Load image paths
        for img_name in sorted(os.listdir(image_dir)):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.image_paths.append(os.path.join(image_dir, img_name))
        
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")
            
        # Initialize camera params from first image
        first_image = cv2.imread(self.image_paths[0])
        if first_image is None:
            raise ValueError(f"Failed to load image: {self.image_paths[0]}")
            
        h, w = first_image.shape[:2]
        self.camera = CameraParameters(w, h, scale_factor)
        
        print(f"Loaded {len(self.image_paths)} images")

    def get_image(self, index):
        """Load and resize an image by index"""
        if index >= len(self.image_paths):
            return None
            
        img = cv2.imread(self.image_paths[index])
        if img is None:
            return None
            
        # Apply downscaling
        return self.resize_image(img)
    
    def resize_image(self, image):
        """Resize image according to scale factor"""
        if image is None:
            return None
            
        # Use pyramidal downsampling for better quality
        factor = self.scale_factor
        while factor > 1:
            image = cv2.pyrDown(image)
            factor /= 2
            
        return image


class FeatureProcessor:
    """Extracts and matches features between images"""
    
    def __init__(self, detector_type='ORB', match_ratio=0.75, feature_count=3000):
        self.match_ratio = match_ratio
        self.feature_count = feature_count
        
        # Initialize feature detector based on type
        if detector_type == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=feature_count)
        elif detector_type == 'SIFT':
            self.detector = cv2.SIFT_create(nfeatures=feature_count)
        elif detector_type == 'AKAZE':
            self.detector = cv2.AKAZE_create()
        else:
            raise ValueError(f"Unsupported detector: {detector_type}")
            
        # Initialize feature matcher
        if detector_type in ['SIFT', 'AKAZE']:
            # L2 norm for floating-point descriptors
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            # Hamming distance for binary descriptors
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            
        self.detector_type = detector_type
    
    def extract_features(self, image):
        """Extract keypoints and descriptors from an image"""
        if image is None:
            return None, None
            
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply contrast enhancement
        gray = cv2.equalizeHist(gray)
        
        # Detect features
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        return keypoints, descriptors
    
    def match_features(self, desc1, desc2):
        """Match features between two images using ratio test"""
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return []
            
        # Apply kNN matching
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < self.match_ratio * n.distance:
                good_matches.append(m)
                
        return good_matches


class MotionEstimator:
    """Estimates camera motion between image pairs"""
    
    def __init__(self, camera_params, ransac_threshold=2.0, min_inliers=8):
        self.camera_params = camera_params
        self.ransac_threshold = ransac_threshold
        self.min_inliers = min_inliers
    
    def estimate_relative_pose(self, points1, points2):
        """Estimate relative pose using essential matrix decomposition"""
        if len(points1) < self.min_inliers or len(points2) < self.min_inliers:
            return None, None, None, 0
            
        # Find the essential matrix
        E, mask = cv2.findEssentialMat(
            points1, points2, self.camera_params.K, 
            method=cv2.RANSAC, 
            prob=0.999, 
            threshold=self.ransac_threshold
        )
        
        if E is None or mask is None:
            return None, None, None, 0
            
        # Count inliers
        inlier_count = np.sum(mask)
        if inlier_count < self.min_inliers:
            return None, None, None, 0
            
        # Filter points to keep only inliers
        inlier_mask = mask.ravel() == 1
        points1_inliers = points1[inlier_mask]
        points2_inliers = points2[inlier_mask]
        
        # Recover pose from essential matrix
        _, R, t, pose_mask = cv2.recoverPose(E, points1_inliers, points2_inliers, self.camera_params.K)
        
        # Filter again based on triangulation check in recoverPose
        valid_mask = pose_mask.ravel() > 0
        
        return R, t, valid_mask, np.sum(valid_mask)
    
    def estimate_pose_pnp(self, points_3d, points_2d):
        """Estimate camera pose from 3D-2D correspondences using PnP"""
        if len(points_3d) < self.min_inliers:
            return None, None, None
            
        # Solve PnP with RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d, points_2d, 
            self.camera_params.K, None,
            iterationsCount=100,
            reprojectionError=self.ransac_threshold,
            flags=cv2.SOLVEPNP_EPNP
        )
        
        if not success or inliers is None or len(inliers) < self.min_inliers:
            return None, None, None
            
        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(rvec)
        
        return R, tvec, inliers


class Triangulator:
    """Handles 3D point triangulation"""
    
    def __init__(self, max_reprojection_error=4.0):
        self.max_reprojection_error = max_reprojection_error
    
    def triangulate_points(self, proj_matrix1, proj_matrix2, points1, points2):
        """Triangulate 3D points from 2D correspondences"""
        # Ensure points are in correct shape for OpenCV
        points1_t = points1.T
        points2_t = points2.T
        
        # Triangulate using OpenCV
        points_4d = cv2.triangulatePoints(proj_matrix1, proj_matrix2, points1_t, points2_t)
        
        # Convert from homogeneous coordinates
        points_3d = (points_4d / points_4d[3]).T
        
        return points_3d[:, :3]  # Return only X, Y, Z
    
    def filter_triangulated_points(self, points_3d, points_2d, proj_matrix, max_error=None):
        """Filter triangulated points based on reprojection error"""
        if max_error is None:
            max_error = self.max_reprojection_error
            
        # Project 3D points back to 2D
        points_3d_homog = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
        points_proj = np.dot(points_3d_homog, proj_matrix.T)
        points_proj = points_proj[:, :2] / points_proj[:, 2:]
        
        # Calculate reprojection errors
        errors = np.linalg.norm(points_proj - points_2d, axis=1)
        
        # Filter based on error threshold
        valid_mask = errors < max_error
        
        return valid_mask, errors
    
    def compute_point_depth(self, point_3d, camera_center, view_direction):
        """Compute the depth of a 3D point relative to a camera"""
        # Vector from camera to point
        vec = point_3d - camera_center
        
        # Project onto view direction
        depth = np.dot(vec, view_direction)
        
        return depth


class ReconstructionTracker:
    """Tracks feature matches across multiple images"""
    
    def __init__(self):
        self.tracks = defaultdict(list)  # Maps track_id to list of (image_idx, feature_idx)
        self.next_track_id = 0
        self.point_cloud = []  # List of (point_3d, color, track_id, error)
        self.colors = []
    
    def add_match(self, image_idx1, feature_idx1, image_idx2, feature_idx2):
        """Add a feature match between two images to tracking"""
        # Look for existing tracks that contain either feature
        track_id1 = self.find_track(image_idx1, feature_idx1)
        track_id2 = self.find_track(image_idx2, feature_idx2)
        
        if track_id1 is None and track_id2 is None:
            # Create new track
            track_id = self.next_track_id
            self.next_track_id += 1
            self.tracks[track_id].append((image_idx1, feature_idx1))
            self.tracks[track_id].append((image_idx2, feature_idx2))
        elif track_id1 is None:
            # Add to track2
            self.tracks[track_id2].append((image_idx1, feature_idx1))
        elif track_id2 is None:
            # Add to track1
            self.tracks[track_id1].append((image_idx2, feature_idx2))
        elif track_id1 != track_id2:
            # Merge tracks
            self.tracks[track_id1].extend(self.tracks[track_id2])
            del self.tracks[track_id2]
    
    def find_track(self, image_idx, feature_idx):
        """Find the track ID that contains a particular feature"""
        for track_id, features in self.tracks.items():
            if (image_idx, feature_idx) in features:
                return track_id
        return None
    
    def get_common_tracks(self, image_idx1, image_idx2):
        """Find tracks visible in both images"""
        common_tracks = []
        
        for track_id, features in self.tracks.items():
            features_in_img1 = [(img, feat) for img, feat in features if img == image_idx1]
            features_in_img2 = [(img, feat) for img, feat in features if img == image_idx2]
            
            if features_in_img1 and features_in_img2:
                common_tracks.append((track_id, features_in_img1[0][1], features_in_img2[0][1]))
                
        return common_tracks
    
    def add_point(self, point_3d, color, track_id, error=0.0):
        """Add a 3D point to the reconstruction"""
        if track_id not in [p[2] for p in self.point_cloud]:
            self.point_cloud.append((point_3d, color, track_id, error))
        
        for i, (p,c,tid,e) in enumerate(self.point_cloud):
            if tid ==track_id:
                updated_color = (c.astype(float) + color.astype(float)) / 2
                    #average of the xe
                self.point_cloud[i] = (point_3d, updated_color.astype(np.uint8), track_id, error)
                break  

    
    def get_track_length(self, track_id):
        """Get the number of observations in a track"""
        return len(self.tracks[track_id])
    
    def get_long_tracks(self, min_length=3):
        """Get tracks that appear in at least min_length images"""
        return [track_id for track_id, features in self.tracks.items() 
                if len(set(img for img, _ in features)) >= min_length]


class PointCloudProcessor:
    """Processes and filters point clouds"""
    
    def __init__(self, clustering_eps=0.3, min_samples=10, scale_factor=200.0):
        self.clustering_eps = clustering_eps
        self.min_samples = min_samples
        self.scale_factor = scale_factor
    
    def filter_statistical_outliers(self, points, std_ratio=2.0):
        """Remove statistical outliers based on distance distribution"""
        if len(points) < 3:
            return points, np.ones(len(points), dtype=bool)
            
        # Compute distances to centroid
        centroid = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        
        # Filter based on standard deviation
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        threshold = mean_dist + std_ratio * std_dist
        
        inlier_mask = distances < threshold
        return points[inlier_mask], inlier_mask
    
    def cluster_points(self, points, colors):
        """Cluster points using DBSCAN to isolate the main object"""
        if len(points) < self.min_samples:
            return points, colors
            
        # Normalize points for better clustering
        scaler = StandardScaler()
        scaled_points = scaler.fit_transform(points)
        
        # Apply DBSCAN
        clustering = DBSCAN(eps=self.clustering_eps, min_samples=self.min_samples)
        labels = clustering.fit_predict(scaled_points)
        
        # Find largest cluster
        valid_labels = labels[labels >= 0]
        if len(valid_labels) == 0:
            return points, colors
            
        unique_labels, counts = np.unique(valid_labels, return_counts=True)
        largest_cluster = unique_labels[np.argmax(counts)]
        
        # Extract points in largest cluster
        cluster_mask = labels == largest_cluster
        return points[cluster_mask], colors[cluster_mask]
    
    def save_ply(self, points, colors, output_path):
        """Save point cloud to PLY file"""
        # Scale points for better visualization
        scaled_points = points * self.scale_factor
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Combine points and colors
        vertices = np.hstack((scaled_points, colors))
        
        # Write PLY header
        header = f'''ply
format ascii 1.0
element vertex {len(vertices)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
        
        # Write data
        with open(output_path, 'w') as f:
            f.write(header)
            for v in vertices:
                f.write(f'{v[0]} {v[1]} {v[2]} {int(v[3])} {int(v[4])} {int(v[5])}\n')
                
        print(f"Saved {len(points)} points to {output_path}")


class MultiViewReconstruction:
    """Main class for multi-view 3D reconstruction"""
    
    def __init__(self, image_dir, output_dir="output", 
                 scale_factor=2.0, feature_type='ORB', 
                 ransac_threshold=2.0, min_track_length=3):
        
        self.output_dir = output_dir
        self.min_track_length = min_track_length
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.images = ImageCollection(image_dir, scale_factor)
        self.features = FeatureProcessor(feature_type, match_ratio=0.75, feature_count=5000)
        self.motion = MotionEstimator(self.images.camera, ransac_threshold)
        self.triangulator = Triangulator()
        self.tracker = ReconstructionTracker()
        self.point_processor = PointCloudProcessor()
        
        # Camera poses (R, t) for each image
        self.poses = []
        
        # First camera is at origin
        self.poses.append((np.eye(3), np.zeros(3)))
        
        # Feature data for each image
        self.image_features = []  # List of (keypoints, descriptors)
    
    def extract_all_features(self):
        """Extract features from all images"""
        print("Extracting features from all images...")
        
        for i in range(len(self.images.image_paths)):
            image = self.images.get_image(i)
            if image is None:
                self.image_features.append((None, None))
                continue
                
            keypoints, descriptors = self.features.extract_features(image)
            self.image_features.append((keypoints, descriptors))
            
            print(f"Image {i}: {len(keypoints) if keypoints is not None else 0} keypoints")
    
    def match_image_pairs(self):
        """Match features between consecutive image pairs"""
        print("Matching features between image pairs...")
        
        # Match consecutive pairs
        for i in range(len(self.images.image_paths) - 1):
            _, desc1 = self.image_features[i]
            _, desc2 = self.image_features[i+1]
            
            if desc1 is None or desc2 is None:
                continue
                
            matches = self.features.match_features(desc1, desc2)
            
            # Add matches to tracker
            for m in matches:
                self.tracker.add_match(i, m.queryIdx, i+1, m.trainIdx)
                
            print(f"Images {i}-{i+1}: {len(matches)} matches")
    
    def initialize_reconstruction(self):
        """Initialize reconstruction with first two images"""
        print("Initializing reconstruction...")
        
        # Get first two valid images
        idx1 = 0
        idx2 = 1
        
        kp1, desc1 = self.image_features[idx1]
        kp2, desc2 = self.image_features[idx2]
        
        if kp1 is None or kp2 is None:
            raise ValueError("Cannot initialize reconstruction with invalid images")
            
        # Match features
        matches = self.features.match_features(desc1, desc2)
        if len(matches) < 8:
            raise ValueError(f"Not enough matches between first two images: {len(matches)}")
            
        # Get matched point coordinates
        pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.array([kp2[m.trainIdx].pt for m in matches])
        
        # Estimate relative pose
        R, t, mask, inlier_count = self.motion.estimate_relative_pose(pts1, pts2)
        
        if R is None or inlier_count < 8:
            raise ValueError(f"Failed to estimate pose between first two images: {inlier_count} inliers")
            
        # Set second camera pose
        self.poses.append((R, t.flatten()))
        
        # Set up projection matrices
        P1 = self.images.camera.get_projection_matrix(np.eye(3), np.zeros(3))
        P2 = self.images.camera.get_projection_matrix(R, t)
        
        # Filter inlier points - FIX: Make sure mask is the right size
        if mask.shape[0] != pts1.shape[0]:
            # Create a mask of the correct size
            full_mask = np.zeros(pts1.shape[0], dtype=bool)
            # Set the first elements to the values from the mask
            full_mask[:mask.shape[0]] = mask.astype(bool).ravel()
            inlier_mask = full_mask
        else:
            inlier_mask = mask.astype(bool).ravel()
        
        pts1_inliers = pts1[inlier_mask]
        pts2_inliers = pts2[inlier_mask]
       
        # Triangulate points
        points_3d = self.triangulator.triangulate_points(P1, P2, pts1_inliers, pts2_inliers)
        
        # Filter points in front of both cameras
        camera1_center = np.zeros(3)
        camera2_center = -R.T @ t.flatten()
        
        valid_points = []
        valid_colors = []
        valid_matches = []
        
        for i, (p3d, p2d1, p2d2) in enumerate(zip(points_3d, pts1_inliers, pts2_inliers)):
            # Check if point is in front of both cameras
            vec1 = p3d - camera1_center
            vec2 = p3d - camera2_center
            
            if vec1[2] > 0 and vec2[2] > 0:
                valid_points.append(p3d)
                
                # Get color from first image
                img1 = self.images.get_image(idx1)
                y, x = int(p2d1[1]), int(p2d1[0])
                if 0 <= y < img1.shape[0] and 0 <= x < img1.shape[1]:
                    color = img1[y, x]
                else:
                    color = np.array([0, 0, 0])

                    
                valid_colors.append(color)
                valid_matches.append(matches[i])
                
                # Add match to tracker
                self.tracker.add_match(idx1, matches[i].queryIdx, idx2, matches[i].trainIdx)
        
        print(f"Initialized with {len(valid_points)} 3D points")
        
        # Add points to reconstruction
        for i, (p3d, color) in enumerate(zip(valid_points, valid_colors)):
            track_id = self.tracker.find_track(idx1, valid_matches[i].queryIdx)
            if track_id is not None:
                self.tracker.add_point(p3d, color, track_id)
    
    def add_next_image(self, image_idx):
        """Add the next image to the reconstruction"""
        if image_idx < 2 or image_idx >= len(self.images.image_paths):
            return False
            
        print(f"Adding image {image_idx} to reconstruction...")
        
        # Get keypoints and descriptors
        kp_next, desc_next = self.image_features[image_idx]
        if kp_next is None:
            return False
            
        # Find 2D-3D correspondences with previous images
        points_3d = []
        points_2d = []
        
        # Look at previous n images (typically 2-3)
        prev_images = range(max(0, image_idx-3), image_idx)
        
        for prev_idx in prev_images:
            # Find tracks visible in both images
            common_tracks = self.tracker.get_common_tracks(prev_idx, image_idx)
            
            for track_id, feat_prev, feat_next in common_tracks:
                # Find if this track has a 3D point
                for point, color, point_track_id, _ in self.tracker.point_cloud:
                    if point_track_id == track_id:
                        # Add the 3D-2D correspondence
                        points_3d.append(point)
                        points_2d.append(kp_next[feat_next].pt)
                        break
        
        if len(points_3d) < 8:
            print(f"Not enough 3D-2D correspondences: {len(points_3d)}")
            return False
            
        # Convert to numpy arrays
        points_3d = np.array(points_3d)
        points_2d = np.array(points_2d)
        
        # Estimate pose using PnP
        R, t, inliers = self.motion.estimate_pose_pnp(points_3d, points_2d)
        
        if R is None:
            print("PnP failed to estimate pose")
            return False
            
        # Set camera pose
        self.poses.append((R, t.flatten()))
        
        # Filter inlier points
        if inliers is not None:
            inlier_indices = inliers.ravel()
            points_3d_inliers = points_3d[inlier_indices]
            points_2d_inliers = points_2d[inlier_indices]
        else:
            points_3d_inliers = points_3d
            points_2d_inliers = points_2d
            
        print(f"Estimated pose with {len(inlier_indices) if inliers is not None else 0} inliers")
        
        # Triangulate new points with previous images
        P_next = self.images.camera.get_projection_matrix(R, t)
        
        # Create new 3D points by triangulating with previous images
        for prev_idx in prev_images:
            # Get keypoints for previous image
            kp_prev, desc_prev = self.image_features[prev_idx]
            if kp_prev is None:
                continue
                
            # Match features
            matches = self.features.match_features(desc_prev, desc_next)
            
            # Get previous camera pose
            R_prev, t_prev = self.poses[prev_idx]
            P_prev = self.images.camera.get_projection_matrix(R_prev, t_prev)
            
            # Get matched points
            pts_prev = []
            pts_next = []
            valid_matches = []
            
            for m in matches:
                # Skip if this correspondence is already part of a 3D point
                track_id = self.tracker.find_track(prev_idx, m.queryIdx)
                if track_id is not None and any(tid == track_id for _, _, tid, _ in self.tracker.point_cloud):
                    continue
                    
                pts_prev.append(kp_prev[m.queryIdx].pt)
                pts_next.append(kp_next[m.trainIdx].pt)
                valid_matches.append(m)
                
                # Add match to tracker
                self.tracker.add_match(prev_idx, m.queryIdx, image_idx, m.trainIdx)
            
            if len(pts_prev) < 8:
                continue
                
            pts_prev = np.array(pts_prev)
            pts_next = np.array(pts_next)
            
            # Triangulate new points
            new_points_3d = self.triangulator.triangulate_points(P_prev, P_next, pts_prev, pts_next)
            
            # Filter new points
            valid_mask, _ = self.triangulator.filter_triangulated_points(new_points_3d[:, :3], pts_next, P_next)
            
            # Add new points to reconstruction
            img_next = self.images.get_image(image_idx)
            
            for i, (is_valid, p3d, p2d, match) in enumerate(zip(valid_mask, new_points_3d, pts_next, valid_matches)):
                if not is_valid:
                    continue
                    
                # Get color from image
                y, x = int(p2d[1]), int(p2d[0])
                if 0 <= y < img_next.shape[0] and 0 <= x < img_next.shape[1]:
                    color = img_next[y, x]
                else:
                    color = np.array([0, 0, 0])
                    
                # Add point to reconstruction
                track_id = self.tracker.find_track(prev_idx, match.queryIdx)
                if track_id is not None:
                    self.tracker.add_point(p3d[:3], color, track_id)
            
            print(f"Added {np.sum(valid_mask)} new points from image pair {prev_idx}-{image_idx}")
        
        return True
    
    def refine_reconstruction(self):
        """Refine the reconstruction by removing outliers"""
        if not self.tracker.point_cloud:
            return
            
        print("Refining reconstruction...")
        
        # Extract points and colors
        points = np.array([p[0] for p in self.tracker.point_cloud])
        colors = np.array([p[1] for p in self.tracker.point_cloud])
        
        # Filter statistical outliers
        points_filtered, stat_mask = self.point_processor.filter_statistical_outliers(points)
        colors_filtered = colors[stat_mask]
        
        print(f"Removed {len(points) - len(points_filtered)} statistical outliers")
        
        # Cluster points
        points_clustered, colors_clustered = self.point_processor.cluster_points(points_filtered, colors_filtered)
        
        print(f"Kept {len(points_clustered)} points after clustering")
        
        # This is where the issue might be - the new point cloud isn't being updated correctly
        new_point_cloud = []
        
        # Instead of trying to match indices, create new point data directly
        for i in range(len(points_clustered)):
            # Create a point with default track ID and error
            new_point = (points_clustered[i], colors_clustered[i], -1, 0.0)
            new_point_cloud.append(new_point)
        
        # Update the tracker's point cloud
        self.tracker.point_cloud = new_point_cloud
        
        # Debug output
        print(f"Final point cloud contains {len(self.tracker.point_cloud)} points")
    
    def run(self):
        """Run the full reconstruction pipeline"""
        # Extract features
        self.extract_all_features()
        
        # Match initial image pairs
        self.match_image_pairs()
        
        # Initialize reconstruction with first pair
        self.initialize_reconstruction()
        
        # Add remaining images incrementally
        for i in range(2, len(self.images.image_paths)):
            success = self.add_next_image(i)
            if not success:
                print(f"Failed to add image {i}")
        
        # Refine the reconstruction
        self.refine_reconstruction()
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save reconstruction results to disk"""
        if not self.tracker.point_cloud:
            print("No points to save")
            return
            
        # Extract points and colors
        points = np.array([p[0] for p in self.tracker.point_cloud])
        colors = np.array([p[1] for p in self.tracker.point_cloud])
        
        # Save point cloud
        output_path = os.path.join(self.output_dir, "reconstruction.ply")
        self.point_processor.save_ply(points, colors, output_path)
        
        # Save camera poses
        poses_file = os.path.join(self.output_dir, "camera_poses.txt")
        with open(poses_file, "w") as f:
            for i, (R, t) in enumerate(self.poses):
                f.write(f"Camera {i}:\n")
                f.write(f"R:\n{R}\n")
                f.write(f"t:\n{t}\n\n")
                
        print(f"Saved {len(points)} points and {len(self.poses)} camera poses")
        
    def visualize_cameras(self, scale=0.1):
        """Visualize camera positions and orientations"""
        # Create figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot camera positions
        for i, (R, t) in enumerate(self.poses):
            # Camera center is -R^T * t
            center = -R.T @ t
            
            # Plot camera center
            ax.scatter(center[0], center[1], center[2], c='red', s=50)
            
            # Plot camera axes
            for j, color in enumerate(['r', 'g', 'b']):
                direction = R.T[:, j] * scale
                ax.quiver(center[0], center[1], center[2],
                         direction[0], direction[1], direction[2],
                         color=color, length=1.0, normalize=True)
                         
            # Add camera label
            ax.text(center[0], center[1], center[2], f"Cam {i}", size=8)
            
        # Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Adjust view
        ax.view_init(elev=20, azim=30)
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, "cameras.png"))
        plt.close()
        
    def visualize_point_cloud(self, max_points=5000):
        """Visualize the reconstructed point cloud"""
        if not self.tracker.point_cloud:
            return
            
        # Extract points and colors
        points = np.array([p[0] for p in self.tracker.point_cloud])
        colors = np.array([p[1] for p in self.tracker.point_cloud]) / 255.0  # Normalize colors
        
        # Subsample if too many points
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
            colors = colors[indices]
        
        # Create figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c=colors, s=1)
                  
        # Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, "point_cloud.png"))
        plt.close()




# Function to capture images from camera
def capture_images(save_dir="data/capture", interval=0.5, max_images=30):
    """Capture images from camera at regular intervals"""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return None
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    image_count = 0
    last_capture_time = time.time()
    
    print("Starting capture... Press 'q' to quit")
    
    while image_count < max_images:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Display frame
        cv2.imshow("Camera Feed", frame)
        
        # Capture frame at interval
        current_time = time.time()
        if current_time - last_capture_time >= interval:
            # Save image with numbered filename
            filename = os.path.join(save_dir, f"image_{image_count:03d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
            
            image_count += 1
            last_capture_time = current_time
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Captured {image_count} images")
    return save_dir if image_count > 0 else None

    
if __name__ == "__main__":
    # Capture images first
    #image_dir = capture_images(save_dir="data/auto_capture", interval=0.5, max_images=30)
    image_dir = ("Data/multipleview")
    if image_dir:
        print(f"Starting 3D reconstruction from {image_dir}...")
        
        # Run reconstruction
        reconstruction = MultiViewReconstruction(
            image_dir=image_dir,
            output_dir="output",
            scale_factor=2.0,
            feature_type="ORB"
        )
        
        reconstruction.run()
        
        # Generate visualizations
        reconstruction.visualize_cameras()
        reconstruction.visualize_point_cloud()
        
        print("Reconstruction complete. Results saved to output")
    else:
        print("Image capture failed.")