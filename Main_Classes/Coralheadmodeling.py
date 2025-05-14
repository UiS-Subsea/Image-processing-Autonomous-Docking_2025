import cv2
import numpy as np
import os
import time
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors



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

class ImageManager:
    """Manages loading and preprocessing of images"""
    
    def __init__(self, image_dir, scale_factor=2.0):
        self.scale_factor = scale_factor
        self.images = []
        self.image_paths = []
        self.camera = None
        image_dir = os.path.normpath(image_dir)  # Normalize path
        for img_name in sorted(os.listdir(image_dir)):    # Load image paths
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.image_paths.append(os.path.join(image_dir, img_name))
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")
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
        return self.resize_image(img)
    
    def resize_image(self, image):
        """Resize image according to scale factor"""
        if image is None:
            return None
        factor = self.scale_factor             # Use pyramidal downsampling for better quality
        while factor > 1:
            image = cv2.pyrDown(image)
            factor /= 2  
        return image


class FeatureProcessor:
    """Extracts and matches features between images"""
    def __init__(self, detector_type='SIFT', match_ratio=0.6, 
                 feature_count=10000, multi_scale=True):
        self.match_ratio = match_ratio
        self.feature_count = feature_count
        self.multi_scale = multi_scale
        
        # Create detector with enhanced parameters
        if detector_type == 'SIFT':
            self.detector = cv2.SIFT_create(
            nfeatures=8000,
            contrastThreshold=0.02, 
            edgeThreshold=20              # Slightly adjusted sigma
            )
        elif detector_type == 'ORB':
            self.detector = cv2.ORB_create(
                nfeatures=feature_count,
                scaleFactor=1.1,         # Finer scale pyramid
                nlevels=12,               # More pyramid levels
                edgeThreshold=15,         # Adjusted edge threshold
                firstLevel=0,
                WTA_K=2,
                scoreType=cv2.ORB_HARRIS_SCORE
            )
        else:
            raise ValueError(f"Unsupported detector: {detector_type}")
        # Matcher with cross-checking
        self.matcher = cv2.BFMatcher()

    def extract_features(self, image):
        """Enhanced feature extraction with quality focus"""
        if image is None:
            return None, None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  # CLAHE for better contrast in textureless regions
        gray = clahe.apply(gray)
        if str(type(self.detector)).find('SIFT') > -1:
            self.detector = cv2.SIFT_create(
                nfeatures=4000,           # n features
                contrastThreshold=0.03,  
                edgeThreshold=15          
            )
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)  
        print(f"Extracted {len(keypoints)} features")
        return keypoints, descriptors
    
    def match_features(self, desc1, desc2):
        """Faster feature matching with FLANN"""
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return []
        FLANN_INDEX_KDTREE = 1
        flann = cv2.FlannBasedMatcher(dict(algorithm=FLANN_INDEX_KDTREE, trees=5),  dict(checks=100) )
        matches = flann.knnMatch(desc1.astype(np.float32), desc2.astype(np.float32), k=2)
        good_matches = []
        for i, (m, n) in enumerate(matches):
            if m.distance < self.match_ratio * n.distance:
                good_matches.append(m)
        print(f"Found {len(good_matches)} good matches out of {len(matches)} total matches")
        return good_matches
    

class MotionEstimator:
    """Estimates camera motion between image pairs"""
    def __init__(self, camera_params, ransac_threshold=0.4, min_inliers=8):
        self.camera_params = camera_params
        self.ransac_threshold = ransac_threshold
        self.min_inliers = min_inliers
    
    def estimate_relative_pose(self, points1, points2):
        """Estimate relative pose using essential matrix decomposition"""
        if len(points1) < self.min_inliers or len(points2) < self.min_inliers:
            return None, None, None, 0
        
        print(f"Estimating pose from {len(points1)} point correspondences")
        E, mask = cv2.findEssentialMat( # # Find the essential matrix
            points1, points2, self.camera_params.K, 
            method=cv2.RANSAC, 
            prob=0.999, 
            threshold=self.ransac_threshold
        )
        if E is None or mask is None:
            return None, None, None, 0
        inlier_count = np.sum(mask)
        print(f"Essential matrix found with {inlier_count} inliers")
        
        if inlier_count < self.min_inliers:
            return None, None, None, 0
        inlier_mask = mask.ravel() == 1
        points1_inliers = points1[inlier_mask]
        points2_inliers = points2[inlier_mask]
        _, R, t, pose_mask = cv2.recoverPose(E, points1_inliers, points2_inliers, self.camera_params.K)
        valid_mask = pose_mask.ravel() > 0   #2nd trangul 
        valid_count = np.sum(valid_mask)
        print(f"Recovered pose with {valid_count} valid points")
        return R, t, valid_mask, valid_count
    
    def estimate_pose_pnp(self, points_3d, points_2d):
        """Estimate camera pose from 3D-2D correspondences using PnP"""
        if len(points_3d) < self.min_inliers:
            return None, None, None
            
        print(f"Estimating PnP pose from {len(points_3d)} 3D-2D correspondences")
        if points_3d.ndim == 2:
            points_3d = points_3d.reshape(-1, 1, 3)
        
        if points_2d.ndim == 2:
            points_2d = points_2d.reshape(-1, 1, 2)
        try:     # Solve PnP with RANSAC
            success, rotationvec, translationvec, inliers = cv2.solvePnPRansac(
                points_3d, points_2d, 
                self.camera_params.K, None,
                iterationsCount=100,
                reprojectionError=self.ransac_threshold,
                flags=cv2.SOLVEPNP_ITERATIVE
            ) 
            if not success or inliers is None or len(inliers) < self.min_inliers:
                print("PnP failed to find a valid pose")
                return None, None, None
            R, _ = cv2.Rodrigues(rotationvec)
            print(f"PnP estimated pose with {len(inliers)} inliers")
            return R, translationvec, inliers  
        except cv2.error as e:
            print(f"PnP error: {str(e)}")
            return None, None, None

class Triangulator:
    """Handles 3D point triangulation"""
    
    def __init__(self, max_reprojection_error=4.0):
        self.max_reprojection_error = max_reprojection_error
    
    def triangulate_points(self, proj_matrix1, proj_matrix2, points1, points2):
        """Triangulate 3D points from 2D correspondences"""
        if points1.shape[0] != 2:  # if Nx2
            points1_t = points1.T
        else:
            points1_t = points1
            
        if points2.shape[0] != 2:  # if Nx2
            points2_t = points2.T
        else:
            points2_t = points2
        #print(f"Triangulating points with shapes: pts1={points1_t.shape}, pts2={points2_t.shape}")
        points_4d = cv2.triangulatePoints(proj_matrix1, proj_matrix2, points1_t, points2_t)
        points_3d = points_4d / points_4d[3]  # Convert from homogeneous coordinates
        valid_mask = points_3d[2] > 0
        max_dist = np.median(np.abs(points_3d[2][valid_mask])) * 5
        valid_mask = valid_mask & (np.abs(points_3d[2]) < max_dist)
        if np.sum(valid_mask) > 0:
            points_3d = points_3d[:, valid_mask]
        return points_3d.T  # Return in shape (N, 4)
    
    def filter_triangulated_points(self, points_3d, points_2d, proj_matrix, max_error=None):
        """Filter triangulated points based on reprojection error"""
        if max_error is None:
            max_error = self.max_reprojection_error
        if points_3d.shape[1] == 4:  # If homogeneous
            points_3d_homog = points_3d
        else:  # If not homogeneous, add column of ones
            points_3d_homog = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
        if points_3d_homog.shape[0] != points_2d.shape[0]: #dimension mismutch
            print(f"Warning: Point count mismatch: 3D points={points_3d_homog.shape[0]}, 2D points={points_2d.shape[0]}")
            min_count = min(points_3d_homog.shape[0], points_2d.shape[0])
            points_3d_homog = points_3d_homog[:min_count]
            points_2d = points_2d[:min_count]
        points_proj = np.dot(points_3d_homog, proj_matrix.T)  # Project 3D points to 2D
        points_proj = points_proj[:, :2] / points_proj[:, 2:]   # Convert to inhomogeneous coordinates
        errors = np.linalg.norm(points_proj - points_2d, axis=1)
        valid_mask = errors < max_error
        print(f"Filtered triangulated points: {np.sum(valid_mask)} valid out of {len(valid_mask)}")
        print(f"Reprojection error stats: min={np.min(errors):.2f}, mean={np.mean(errors):.2f}, max={np.max(errors):.2f}")
        
        return valid_mask, errors
    
    def compute_reprojection_error(self, points_3d, image_points, transform_matrix, K, homogeneous=True):
        """Calculate reprojection error for 3D points"""
        rot_matrix = transform_matrix[:3, :3]
        tran_vector = transform_matrix[:3, 3]
        rot_vector, _ = cv2.Rodrigues(rot_matrix)
        if homogeneous:
            obj_points = cv2.convertPointsFromHomogeneous(points_3d)
        else:
            obj_points = points_3d.reshape(-1, 1, 3)
        image_points_calc, _ = cv2.projectPoints(obj_points, rot_vector, tran_vector, K, None)
        image_points_calc = np.float32(image_points_calc[:, 0, :])
        total_error = cv2.norm(image_points_calc, np.float32(image_points), cv2.NORM_L2)
        return total_error / len(image_points_calc), obj_points

class PointCloudProcessor:
    """Processes and filters point clouds"""
    
    def __init__(self, clustering_eps=0.3, min_samples=5, scale_factor=200.0):
        self.clustering_eps = clustering_eps
        self.min_samples = min_samples
        self.scale_factor = scale_factor
    
    def calculate_dimensions(self, points, scale_factor = 0.05):
        """Calculate diameter and height of the point cloud"""
        if len(points) < 10:
            print("Not enough points to calculate dimensions")
            return 0.0, 0.0 
        pca = PCA(n_components=3)
        pca.fit(points)
        axes = pca.components_
        projected = pca.transform(points)
        vertical_idx = np.argmin(pca.explained_variance_)
        horizontal_indices = [i for i in range(3) if i != vertical_idx]
        
        # Extract planar points (for diameter calculation)
        planar_points = projected[:, horizontal_indices]
        planar_distances = pdist(planar_points)
        if len(planar_distances) > 0:
            diameter = np.max(planar_distances) * scale_factor
        else:
            diameter = 0.0
        height = (np.max(projected[:, vertical_idx]) - np.min(projected[:, vertical_idx])) * scale_factor
        print(f"Calculated dimensions - Diameter: {diameter:.2f}, Height: {height:.2f}")
        return diameter, height
    
    def detect_diseased_area(self, points, colors, 
                            threshold_r=160, threshold_g=70, threshold_b=80,
                            scale_to_cm=0.1):
        """Detect diseased areas based on color thresholds and calculate actual area"""
        if len(points) < 3:
            return None, 0.0
        diseased_mask = (colors[:, 0] > threshold_r) & (colors[:, 1] < threshold_g) & (colors[:, 2] < threshold_b)
        if np.sum(diseased_mask) > 8:
            diseased_points = points[diseased_mask]
          
            nn = NearestNeighbors(n_neighbors=2).fit(diseased_points)
            distances, _ = nn.kneighbors(diseased_points)
            avg_distance = np.mean(distances[:, 1])
            
            # Calculate actual area without capping
            diseased_area = np.sum(diseased_mask) * avg_distance * avg_distance * scale_to_cm
            
            # Print diagnostics
            print(f"Detected {np.sum(diseased_mask)} diseased points")
            print(f"Average point distance: {avg_distance:.4f}")
            print(f"Calculated diseased area: {diseased_area:.2f} cm²")
            
            return diseased_points, diseased_area
        else:
            print("Insufficient diseased points detected")
            return None, 0.0  # Return zero if no disease detected
    
    def save_ply(self, points, colors, output_path):
        """Save point cloud to PLY file"""
        out_points = points * self.scale_factor # Scale points for better visualization
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        verts = np.hstack((out_points, colors))
        # Additional filtering based on distance from mean
        mean = np.mean(verts[:, :3], axis=0)
        scaled_verts = verts[:, :3] - mean
        dist = np.sqrt(np.sum(scaled_verts ** 2, axis=1))
        indx = np.where(dist < np.mean(dist) + 300)
        verts = verts[indx]
        
        # Write PLY header
        ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar blue
property uchar green
property uchar red
end_header
'''
        # Write data
        with open(output_path, 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f %d %d %d')
                
        print(f"Saved {len(verts)} points to {output_path}")

class ReconstructionTracker:
    """Tracks feature matches across multiple images"""
       
    def __init__(self):
        self.tracks = defaultdict(list)  # Maps track_id to list of (image_idx, feature_idx)
        self.next_track_id = 0
        self.point_cloud = []  # List of (point_3d, color, track_id, error)
        
        # Add index structures for faster lookups
        self.feature_to_track = {}  # Maps (image_idx, feature_idx) to track_id
        self.image_to_tracks = defaultdict(set)  # Maps image_idx to set of track_ids
    def add_match(self, image_idx1, feature_idx1, image_idx2, feature_idx2):
        """Add a feature match between two images to tracking with faster lookups"""
        # Check existing tracks using hash lookups instead of iteration
        key1 = (image_idx1, feature_idx1)
        key2 = (image_idx2, feature_idx2)
        
        track_id1 = self.feature_to_track.get(key1)
        track_id2 = self.feature_to_track.get(key2)
        
        if track_id1 is None and track_id2 is None:
            # Create new track
            track_id = self.next_track_id
            self.next_track_id += 1
            
            self.tracks[track_id].append(key1)
            self.tracks[track_id].append(key2)
            
            # Update indices
            self.feature_to_track[key1] = track_id
            self.feature_to_track[key2] = track_id
            self.image_to_tracks[image_idx1].add(track_id)
            self.image_to_tracks[image_idx2].add(track_id)
            
        elif track_id1 is None:
            # Add to track2
            self.tracks[track_id2].append(key1)
            self.feature_to_track[key1] = track_id2
            self.image_to_tracks[image_idx1].add(track_id2)
            
        elif track_id2 is None:
            # Add to track1
            self.tracks[track_id1].append(key2)
            self.feature_to_track[key2] = track_id1
            self.image_to_tracks[image_idx2].add(track_id1)
            
        elif track_id1 != track_id2:
            # Merge tracks (keep the smaller track_id)
            if track_id1 < track_id2:
                keep_id, remove_id = track_id1, track_id2
            else:
                keep_id, remove_id = track_id2, track_id1
            
            # Update all features from the removed track
            for img_idx, feat_idx in self.tracks[remove_id]:
                self.feature_to_track[(img_idx, feat_idx)] = keep_id
                self.image_to_tracks[img_idx].add(keep_id)
                self.image_to_tracks[img_idx].discard(remove_id)
                self.tracks[keep_id].append((img_idx, feat_idx))
            
            # Remove the old track
            del self.tracks[remove_id]
    
    def find_track(self, image_idx, feature_idx):
        """Find track using dictionary lookup - O(1) instead of O(n)"""
        return self.feature_to_track.get((image_idx, feature_idx))


    def add_point(self, point_3d, color, track_id, error=0.0):
        if point_3d is None or color is None or track_id is None:
            print(f"Invalid point addition: point={point_3d}, color={color}, track_id={track_id}")
            return False
    
        # Convert point to correct format if needed
        if isinstance(point_3d, np.ndarray) and point_3d.shape[0] == 4:  # Homogeneous
            point_3d = point_3d[:3] / point_3d[3]  # Convert to 3D
        
        # Check for existing point with same track ID
        for i, (p, c, tid, e) in enumerate(self.point_cloud):
            if tid == track_id:
                # Update existing point
                updated_color = (c.astype(float) + color.astype(float)) / 2
                self.point_cloud[i] = (point_3d, updated_color.astype(np.uint8), track_id, error)
                return True
        
        # Add new point
        self.point_cloud.append((point_3d, color, track_id, error))
        
        # Debug print
        print(f"Added point for track {track_id}: 3D={point_3d}, Color={color}")
        return True


class MVS:
    """Main class for multi-view 3D reconstruction"""
    
    def __init__(self, image_dir="Data", output_dir="output", 
                 scale_factor=2.0, feature_type='SIFT', 
                 ransac_threshold=0.4, min_track_length=3,  skip_image_loading = False):
        
        self.output_dir = output_dir
        self.min_track_length = min_track_length
        self.reprojection_errors = []
        self.image_dir = image_dir 
        self.scale_factor = scale_factor 
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
    
        
        # Initialize components
        if not skip_image_loading:
            self.images = ImageManager(image_dir, scale_factor)
            self.motion = MotionEstimator(self.images.camera, ransac_threshold)
            self.triangulator = Triangulator(max_reprojection_error=7.0)
        else:
            # Create a dummy camera for capture_images
            self.images = None
            self.motion = None
            self.bundle_adjuster = None
        self.features = FeatureProcessor(feature_type, match_ratio=0.6, feature_count=5000, multi_scale=True)
        self.tracker = ReconstructionTracker()
        self.point_processor = PointCloudProcessor(clustering_eps=0.3, min_samples=4)
        
        # Camera poses (R, t) for each image
        self.poses = []
        self.poses.append((np.eye(3), np.zeros(3))) #global origin with identity rotation and zero translation
        self.image_features = []  # List of (keypoints, descriptors)
        self.total_points = np.zeros((1, 3))
        self.total_colors = np.zeros((1, 3))

    def capture_images(self, interval=0.5, max_images=20):
            """Capture images from the camera at regular intervals and save them for reconstruction."""
            cap = cv2.VideoCapture(0)
            if not cap.isOpened(): 
                print("Error: Could not open camera")
                return None
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Camera Feed', 800, 600)
            image_count, last_capture_time = 0, time.time()
            while image_count < max_images:
                ret, frame = cap.read()
                if not ret: break
                cv2.imshow('Camera Feed', frame)
                if time.time() - last_capture_time >= interval:
                    filename = os.path.join(self.image_dir, f"image_{image_count:03d}.jpg")
                    cv2.imwrite(filename, frame)
                    print(f"Saved {filename}")
                    image_count += 1
                    last_capture_time = time.time()
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            cap.release()
            cv2.destroyAllWindows()
            return self.image_dir if image_count else None
    
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
        """Optimized matching strategy that limits the number of matches"""
        print("Matching features between image pairs...")
        
        # Only match sequential pairs and a few non-sequential pairs
        for i in range(len(self.images.image_paths) - 1):
            # Match sequential pairs
            keypoints1, desc1 = self.image_features[i]
            keypoints2, desc2 = self.image_features[i+1]
            
            if desc1 is None or desc2 is None:
                continue
            
            matches = self.features.match_features(desc1, desc2)
            
            # Add matches to tracker
            for m in matches:
                self.tracker.add_match(i, m.queryIdx, i+1, m.trainIdx)
            
            print(f"Images {i}-{i+1}: {len(matches)} matches")
            if i+3 < len(self.images.image_paths):
                keypoints3, desc3 = self.image_features[i+3]
                
                if desc3 is None:
                    continue
                matches = self.features.match_features(desc1, desc3)
                # Add matches to tracker
                for m in matches:
                    self.tracker.add_match(i, m.queryIdx, i+3, m.trainIdx) 
                print(f"Images {i}-{i+3}: {len(matches)} matches")
        
    def initialize_reconstruction(self):
        """Enhanced initialization with proper point creation"""
        print("Initializing reconstruction...")
        idx1, idx2 = 0, 1
        kp1, desc1 = self.image_features[idx1]
        kp2, desc2 = self.image_features[idx2]
        if kp1 is None or kp2 is None:
            raise ValueError("Cannot initialize reconstruction with invalid images")
        matches = self.features.match_features(desc1, desc2)
        if len(matches) < 20:
            raise ValueError("Not enough matches for initialization")
        pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.array([kp2[m.trainIdx].pt for m in matches])
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.images.camera.K, 
            method=cv2.RANSAC, 
            prob=0.999, 
            threshold=1.0  # Stricter threshold
        )
        if E is None or mask is None:
            raise ValueError("Could not estimate essential matrix")
        inlier_count = np.sum(mask)
        print(f"Essential matrix found with {inlier_count} inliers")
        
        if inlier_count < 8:
            raise ValueError("Not enough inliers for pose estimation")
        inlier_mask = mask.ravel() == 1
        pts1_inliers = pts1[inlier_mask]
        pts2_inliers = pts2[inlier_mask]
        _, R, t, pose_mask = cv2.recoverPose(E, pts1_inliers, pts2_inliers, self.images.camera.K)
        self.poses.append((R, t.flatten()))
        P1 = self.images.camera.get_projection_matrix(np.eye(3), np.zeros(3))
        P2 = self.images.camera.get_projection_matrix(R, t)
        points_3d = self.triangulator.triangulate_points(P1, P2, pts1_inliers.T, pts2_inliers.T)
        valid_mask, _ = self.triangulator.filter_triangulated_points(       
        # Filter points with negative depth or large reprojection error
            points_3d, pts2_inliers, P2, max_error=2.0)
        img1 = self.images.get_image(idx1) #gives u info color
        img2 = self.images.get_image(idx2)
        if img1 is None or img2 is None:
            raise ValueError("Could not load images for color extraction")
        count = 0
        for i, valid in enumerate(valid_mask):
            if not valid:
                continue
            p3d = points_3d[i]
            # retr 2D coordinates and check if they're within image bounds
            x1, y1 = int(pts1_inliers[i][0]), int(pts1_inliers[i][1])
            if 0 <= x1 < img1.shape[1] and 0 <= y1 < img1.shape[0]:
                # retr color from image
                color = img1[y1, x1]
                # Find match index in original array to get the track ID and Find which match this corresponds to
                match_idx = None
                for j, m in enumerate(matches):
                    if inlier_mask[j] and i == sum(inlier_mask[:j]):
                        match_idx = j
                        break         
                if match_idx is not None:
                    self.tracker.add_match(idx1, matches[match_idx].queryIdx, idx2, matches[match_idx].trainIdx)
                    track_id = self.tracker.find_track(idx1, matches[match_idx].queryIdx)
                    if track_id is not None:
                        self.tracker.add_point(p3d, color, track_id)
                        if len(self.total_points) <= 1:  # Just the initial empty row
                            self.total_points = np.vstack((self.total_points, p3d[:3].reshape(1, 3)))
                            self.total_colors = np.vstack((self.total_colors, color.reshape(1, 3)))
                        else:
                            self.total_points = np.vstack((self.total_points, p3d[:3].reshape(1, 3)))
                            self.total_colors = np.vstack((self.total_colors, color.reshape(1, 3)))
                        
                        count += 1
        
        print(f"Initialized reconstruction with {count} 3D points")
        
        if count < 10:
            raise ValueError("Failed to create enough initial 3D points")
    
    def add_next_image(self, image_idx):
        """Enhanced image addition with proper point creation"""
        if image_idx < 2 or image_idx >= len(self.images.image_paths):
            return False
        kp_next, desc_next = self.image_features[image_idx]
        if kp_next is None or desc_next is None:
            print("No keypoints or descriptors for this image")
            return False
        # Try matching with multiple previous images
        prev_images = range(max(0, image_idx-5), image_idx)
        all_points_3d = []
        all_points_2d = []
        all_track_ids = []
        all_match_details = []
        
        for prev_idx in prev_images:
            kp_prev, desc_prev = self.image_features[prev_idx]
            if kp_prev is None or desc_prev is None:
                continue
            matches = self.features.match_features(desc_prev, desc_next)
            for match in matches:
                track_id = self.tracker.find_track(prev_idx, match.queryIdx)
                if track_id is not None:
                    #  find 3D point for this track
                    for point, color, point_track_id, _ in self.tracker.point_cloud:
                        if point_track_id == track_id:
                            # Add 3D-2D correspondence
                            all_points_3d.append(point)
                            all_points_2d.append(kp_next[match.trainIdx].pt)
                            all_track_ids.append(track_id)
                            all_match_details.append({
                                'prev_idx': prev_idx,
                                'prev_kp_idx': match.queryIdx,
                                'curr_kp_idx': match.trainIdx
                            })
                            break
        print(f"Found {len(all_points_3d)} 3D-2D correspondences")
        if len(all_points_3d) < 8:
            print(f"Not enough 3D-2D correspondences: {len(all_points_3d)}")
            # If not enough correspondences but we have previous camera poses,
            # estimate pose relative to the previous camera
            if image_idx > 2 and len(self.poses) >= image_idx:
                R_prev, t_prev = self.poses[image_idx-1]
                kp_prev, desc_prev = self.image_features[image_idx-1]
                matches = self.features.match_features(desc_prev, desc_next)
                if len(matches) >= 15:  # Need enough matches for essential matrix
                    pts_prev = np.array([kp_prev[m.queryIdx].pt for m in matches])
                    pts_next = np.array([kp_next[m.trainIdx].pt for m in matches])
                    E, mask = cv2.findEssentialMat(
                        pts_prev, pts_next, self.images.camera.K,
                        method=cv2.RANSAC, prob=0.999, threshold=1.0
                    )
                    
                    if E is not None and np.sum(mask) >= 8:
                        _, R_rel, t_rel, _ = cv2.recoverPose(
                            E, pts_prev[mask.ravel() == 1], pts_next[mask.ravel() == 1], 
                            self.images.camera.K
                        )
                        R_next = R_rel @ R_prev
                        t_next = R_rel @ t_prev + t_rel.flatten()
                        self.poses.append((R_next, t_next))
                        P_prev = self.images.camera.get_projection_matrix(R_prev, t_prev)
                        P_next = self.images.camera.get_projection_matrix(R_next, t_next)
                        
                        # Triangulate new points
                        points_3d = self.triangulator.triangulate_points(
                            P_prev, P_next, pts_prev.T, pts_next.T
                        )
                        valid_mask, _ = self.triangulator.filter_triangulated_points(
                            points_3d, pts_next, P_next, max_error=2.0
                        )
                        img_next = self.images.get_image(image_idx)
                        point_count = 0
                        for i, valid in enumerate(valid_mask):
                            if not valid:
                                continue
                            self.tracker.add_match(
                                image_idx-1, matches[i].queryIdx, 
                                image_idx, matches[i].trainIdx
                            )
                            track_id = self.tracker.find_track(image_idx-1, matches[i].queryIdx)
                            
                            if track_id is not None:
                                x, y = int(pts_next[i][0]), int(pts_next[i][1])
                                color = img_next[y, x] if (0 <= x < img_next.shape[1] and 0 <= y < img_next.shape[0]) else np.array([0, 0, 0])
                                p3d = points_3d[i][:3] if points_3d[i].shape[0] == 4 else points_3d[i]
                                self.tracker.add_point(p3d, color, track_id)
                                self.total_points = np.vstack((self.total_points, p3d.reshape(1, 3)))
                                self.total_colors = np.vstack((self.total_colors, color.reshape(1, 3)))
                                point_count += 1
                        
                        print(f"Added {point_count} new points via relative pose estimation")
                        return True
            
            return False
        points_3d = np.array(all_points_3d)
        points_2d = np.array(all_points_2d)
        points_3d_for_pnp = points_3d.reshape(-1, 1, 3)
        points_2d_for_pnp = points_2d.reshape(-1, 1, 2)
        success, rotationvec, translationvec, inliers = cv2.solvePnPRansac(
            points_3d_for_pnp, points_2d_for_pnp, 
            self.images.camera.K, None,
            iterationsCount=100,
            reprojectionError=2.0,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success or inliers is None or len(inliers) < 8:
            print("PnP failed to estimate pose")
            return False
        R, _ = cv2.Rodrigues(rotationvec)  # Converts rotation vector to matrix        
        t = translationvec.flatten()
        self.poses.append((R, t))
        P_next = self.images.camera.get_projection_matrix(R, t)
        print(f"Estimated pose with {len(inliers)} inliers")
        valid_pts_3d = points_3d_for_pnp[inliers[:, 0]]
        valid_pts_2d = points_2d_for_pnp[inliers[:, 0]]
        if valid_pts_3d.shape[1] == 1:
            valid_pts_3d = valid_pts_3d.reshape(-1, 3)
        if valid_pts_2d.shape[1] == 1:
            valid_pts_2d = valid_pts_2d.reshape(-1, 2)

        error, _ = self.triangulator.compute_reprojection_error(
            valid_pts_3d, valid_pts_2d, 
            np.hstack((R, t.reshape(3, 1))), 
            self.images.camera.K, 
            homogeneous=False
        )
        self.reprojection_errors.append(error)
        plt.plot(image_idx, error, 'bo', markersize=8)
        plt.draw() 
        img_next = self.images.get_image(image_idx)
        for prev_idx in prev_images:
            if prev_idx >= len(self.poses):          #  previous camera pose and projection matrix
                continue
            R_prev, t_prev = self.poses[prev_idx]
            P_prev = self.images.camera.get_projection_matrix(R_prev, t_prev)
            kp_prev, desc_prev = self.image_features[prev_idx]
            if kp_prev is None:
                continue
            matches = self.features.match_features(desc_prev, desc_next)
            pts_prev = []
            pts_next = []
            valid_matches = []
            for m in matches:  
                # Skip if this correspondence is already part of a 3D point
                track_id = self.tracker.find_track(prev_idx, m.queryIdx)
                if track_id is not None and any(tid == track_id for point, color, tid, _ in self.tracker.point_cloud):
                    continue
                    
                pts_prev.append(kp_prev[m.queryIdx].pt)
                pts_next.append(kp_next[m.trainIdx].pt)
                valid_matches.append(m)
                self.tracker.add_match(prev_idx, m.queryIdx, image_idx, m.trainIdx)
            
            if len(pts_prev) < 8:
                continue
            pts_prev = np.array(pts_prev)
            pts_next = np.array(pts_next)
            new_points_3d = self.triangulator.triangulate_points(P_prev, P_next, pts_prev.T, pts_next.T)
            valid_mask, errors = self.triangulator.filter_triangulated_points(
                new_points_3d, pts_next, P_next)
            point_count = 0
            for i, (is_valid, p3d, p2d, match) in enumerate(zip(valid_mask, new_points_3d, pts_next, valid_matches)):
                if not is_valid:
                    continue
                    
                # Convert homogeneous to 3D if needed
                if p3d.shape[0] == 4:
                    p3d_3d = p3d[:3] / p3d[3]
                else:
                    p3d_3d = p3d
                
                # retrive color from image
                y, x = int(p2d[1]), int(p2d[0])
                if 0 <= y < img_next.shape[0] and 0 <= x < img_next.shape[1]:
                    color = img_next[y, x]
                else:
                    color = np.array([0, 0, 0])
                    
                # Add point to reconstruction
                track_id = self.tracker.find_track(prev_idx, match.queryIdx)
                if track_id is not None:
                    if self.tracker.add_point(p3d_3d, color, track_id, errors[i]):
                        # Add to total points and colors
                        self.total_points = np.vstack((self.total_points, p3d_3d.reshape(1, 3)))
                        self.total_colors = np.vstack((self.total_colors, color.reshape(1, 3)))
                        point_count += 1
            
            print(f"Added {point_count} new points from image pair {prev_idx}-{image_idx}")
        
        return True
    def save_error_plot(self):
        """Save the reprojection error plot"""
        plt.figure(figsize=(10, 6))
        plt.title("Reprojection Errors")
        plt.xlabel("Image Index")
        plt.ylabel("Reprojection Error")
        plt.grid(True)
        
        if len(self.reprojection_errors) > 0:
            print(f"Saving {len(self.reprojection_errors)} reprojection errors: {self.reprojection_errors}")
            image_indices = range(2, 2 + len(self.reprojection_errors))
            plt.plot(image_indices, self.reprojection_errors, 'bo-', markersize=8)
            plt.ylim(0, max(self.reprojection_errors) * 1.2)  # Set y axis limits
        else:
            print("No reprojection errors to plot")
        
        plt.savefig(os.path.join(self.output_dir, "reprojection_errors.png"))
        plt.close()

    def save_point_cloud_to_ply(self):
        """Process the final point cloud"""
        if self.total_points.shape[0] <= 1:
            print("No points to process")
            return
        self.total_points = self.total_points[1:]
        self.total_colors = self.total_colors[1:]
        output_path = os.path.join(self.output_dir, "reconstruction.ply")
        self.point_processor.save_ply(self.total_points, self.total_colors, output_path)
            
    def run(self, capture_new_images=True, interval=0.5, max_images=60):
        """Run the full reconstruction pipeline and return useful information"""
        try:
            if capture_new_images:
                #print("Starting camera capture...")
                #image_dir = self.capture_images(interval=interval, max_images=max_images)
                image_dir = ("../Data1/Images")
                
                if not os.path.exists(image_dir):
                    return {
                        'success': False,
                        'error': "Failed to capture images or no images were captured."
                    }
                
                # Initialize components with captured images
                self.images = ImageManager(image_dir, self.scale_factor)
                self.motion = MotionEstimator(self.images.camera, 0.4)
                self.triangulator = Triangulator(max_reprojection_error=7.0)
            self.extract_all_features()
            self.match_image_pairs()
            self.initialize_reconstruction()
            
            plt.figure(figsize=(10, 6))
            plt.title("Reprojection Errors")
            plt.xlabel("Image Index")
            plt.ylabel("Reprojection Error")
            plt.xlim(0, len(self.images.image_paths))  # Set x-axis range
            plt.ylim(0, 10)  # Start with a reasonable y-axis range
            plt.grid(True)
            plt.ion()
            for i in tqdm(range(2, len(self.images.image_paths))):      
                success = self.add_next_image(i) # add remaining images incrementally
                if success:
                    plt.draw()  # Refresh the plot
                    plt.pause(0.1) 
                else:
                    print("failed")
            self.save_point_cloud_to_ply()
            self.save_camera_poses()
            self.save_error_plot()
            self.visualize_cameras()
            self.visualize_point_cloud()
            # Calculate dimensions if we have points
            if hasattr(self, 'total_points') and self.total_points.shape[0] > 1:
                points = self.total_points[1:]  # Skip initial zero row
                colors = self.total_colors[1:]
                diameter, height = self.point_processor.calculate_dimensions(points)
                diseased_points, diseased_area = self.point_processor.detect_diseased_area(points, colors)
                results = {
                    'success': True,
                    'point_cloud_path': os.path.join(self.output_dir, "reconstruction.ply"),
                    'visualization_path': os.path.join(self.output_dir, "coral_measurements.png"),
                    'num_points': points.shape[0],
                    'diameter_cm': diameter,
                    'height_cm': height,
                    'diseased_area_cm2': diseased_area,
                    'diseased_percentage': (diseased_area / (np.pi * (diameter/2)**2)) * 100 if diameter > 0 else 0,
                    'reprojection_errors': self.reprojection_errors,
                    'camera_poses': self.poses
                }
                
                return results
            return {
            'success': False,
            'error': "No points reconstructed"
            }
        
        except Exception as e:
            print(f"Error during reconstruction: {str(e)}")
            import traceback
            traceback.print_exc()
            cv2.destroyAllWindows()
            
            # Return error information
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def save_camera_poses(self):
        """Save camera poses to file"""
        poses_file = os.path.join(self.output_dir, "camera_poses.txt")
        with open(poses_file, "w") as f:
            for i, (R, t) in enumerate(self.poses):
                f.write(f"Camera {i}:\n")
                f.write(f"R:\n{R}\n")
                f.write(f"t:\n{t}\n\n")
        pose_array = np.hstack((self.images.camera.K.ravel(),))
        for R, t in self.poses:
            proj_matrix = self.images.camera.get_projection_matrix(R, t)
            pose_array = np.hstack((pose_array, proj_matrix.ravel()))  
        np.savetxt(os.path.join(self.output_dir, "pose_array.csv"), pose_array, delimiter='\n')
        print(f"Saved {len(self.poses)} camera poses")
        
    def visualize_cameras(self, scale=0.1):
        """Visualize camera positions and orientations"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for i, (R, t) in enumerate(self.poses):
            # Camera center is -R^T * t
            center = -R.T @ t #@ matrix multip or use np.matmul()
            ax.scatter(center[0], center[1], center[2], c='red', s=50)
            # Plot camera axes
            for j, color in enumerate(['r', 'g', 'b']):
                direction = R.T[:, j] * scale
                ax.quiver(center[0], center[1], center[2],
                         direction[0], direction[1], direction[2],
                         color=color, length=1.0, normalize=True)
            ax.text(center[0], center[1], center[2], f"Cam {i}", size=8)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=20, azim=30)
        plt.savefig(os.path.join(self.output_dir, "cameras.png"))
        plt.close()
        
    def visualize_point_cloud(self, max_points=5000):
        """Visualize the reconstructed point cloud with measurements"""
        if self.total_points.shape[0] <= 1:
            return
        if hasattr(self, 'segmented_points') and hasattr(self, 'segmented_colors'):
            points = self.segmented_points
            colors = self.segmented_colors
        else:
            points = self.total_points[1:]
            colors = self.total_colors[1:]
        diameter, height = self.point_processor.calculate_dimensions(points)
        diseased_points, diseased_area = self.point_processor.detect_diseased_area(points, colors)
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
            colors = colors[indices]
        fig = plt.figure(figsize=(16, 14))
        ax = fig.add_subplot(111, projection='3d')
        colors_rgb = colors[:, ::-1]  # Reverse color channels (BGR to RGB)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
           c=colors_rgb/255.0, s=50)
        ax.view_init(elev=30, azim=45)
        if diseased_points is not None and len(diseased_points) > 0:
            ax.scatter(diseased_points[:, 0], diseased_points[:, 1], diseased_points[:, 2],
                    color='red', s=20, alpha=0.7)
        # Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        measurement_text = f"Diameter: {diameter:.2f} cm\n"
        measurement_text += f"Height: {height:.2f} cm\n"
        measurement_text += f"Diseased Area: {diseased_area:.2f} cm²"
        
        ax.text2D(0.05, 0.95, measurement_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        plt.savefig(os.path.join(self.output_dir, "coral_measurements.png"))
        plt.savefig(os.path.join(self.output_dir, "point_cloud.png"))
        plt.close()
        print(f"Coral Measurements:")
        print(f"Diameter: {diameter:.2f} cm")
        print(f"Height: {height:.2f} cm")
        print(f"Diseased Area: {diseased_area:.2f} cm²")

    
if __name__ == "__main__":
    reconstruction = MVS(skip_image_loading=True)
    results = reconstruction.run(capture_new_images=True)
    
    if results['success']:
        print(f"Reconstruction complete. Results saved to {reconstruction.output_dir}")
    else:
        print(f"Reconstruction failed: {results['error']}")