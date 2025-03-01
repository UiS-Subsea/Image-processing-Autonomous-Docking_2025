
import numpy as np
import time
import math
#from cv2 import aruco
import cv2 as cv


class AutonomousDocking:
    def __init__(self):
        self.driving_data = [0, 0, 0, 0, 0, 0, 0, 0]
        self.frame = None
        self.down_frame = None
        self.draw_grouts = False
        self.draw_grout_boxes = True
        self.angle_good = False
        self.aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_params = cv.aruco.DetectorParameters()

        #Added this for docking verification
        self.docking_start_time = None
        self.is_docking = False
        self.docking_positions = []
        self.POSITION_HISTORY_SIZE = 10  # Store last 10 positions
        self.REQUIRED_STABLE_TIME = 10  # seconds
        self.STABILITY_THRESHOLD = 20  # pixels
    def verify_stable_docking(self, current_position):
        """
        Verify if the vehicle maintains a stable position for 10 seconds
        Returns: bool indicating if docking is stable
        """
        current_time = time.time()
        
        # If not currently in docking state, start docking
        if not self.is_docking:
            self.is_docking = True
            self.docking_start_time = current_time
            self.docking_positions = []
            return False

        # Add current position to history
        self.docking_positions.append(current_position)
        if len(self.docking_positions) > self.POSITION_HISTORY_SIZE:
            self.docking_positions.pop(0)

        # Check if position is stable
        is_stable = self.check_position_stability()
        
        # If position is not stable, reset docking
        if not is_stable:
            self.is_docking = False
            return False
        if current_time - self.docking_start_time >= self.REQUIRED_STABLE_TIME:
            print(f"Stable docking achieved for {self.REQUIRED_STABLE_TIME} seconds!")
            return True

        return False
    
    def check_position_stability(self):
        """
        Check if the vehicle position is stable within threshold
        Returns: bool indicating if position is stable
        """
        if len(self.docking_positions) < 2:
            return True

        # Check maximum deviation in x and y coordinates
        x_coords = [pos[0] for pos in self.docking_positions]
        y_coords = [pos[1] for pos in self.docking_positions]
        
        x_deviation = max(x_coords) - min(x_coords)
        y_deviation = max(y_coords) - min(y_coords)

        return (x_deviation < self.STABILITY_THRESHOLD and 
                y_deviation < self.STABILITY_THRESHOLD)
    
    def run(self, front_frame, down_frame):
        self.frame = front_frame
        self.down_frame = down_frame
        self.update()
        data = self.get_driving_data()
        
        return self.frame, self.down_frame, data

    def update(self):
        # First check orientation using floor markers
        self.rotation_commands()

        if self.angle_good:
            frame_width = self.frame.shape[1]
            frame_height = self.frame.shape[0]
            
            frame_centerpoint = (frame_width // 2, frame_height // 2)
            docking_position = self.find_docking_station()
            
            if docking_position == (0, 0):
                print("No docking station found!") 
                return "No docking station found!"
            # Added stable docking verification
            if self.verify_stable_docking(docking_position):
                self.driving_data = [40, [0, 0, 0, 0, 0, 0, 0, 0]]
                raise SystemExit  # Docking complete
            
            width_diff = frame_centerpoint[0] - docking_position[0]
            height_diff = frame_centerpoint[1] - docking_position[1]
            percent_width_diff = (width_diff / frame_width) * 100
            percent_height_diff = (height_diff / frame_height) * 100
            
            # Calculate distance to docking station using ArUco marker size
            distance = self.calculate_distance(docking_position)
            angle = self.find_relative_angle() 
            print(f"Angle: {angle}, Distance: {distance}, Driving Data: {self.driving_data}")

            
            if distance < 0.3:  # If closer than 30cm
                
                self.driving_data = [40, [0, 0, 0, 0, 0, 0, 0, 0]]
                raise SystemExit  # Docking complete
            else:
                self.driving_data = self.regulate_position(percent_width_diff, percent_height_diff)

    def detect_aruco_markers(self, frame=None):
        """Detect ArUco markers in the given frame"""
        # Use the frame provided or use self.frame
        if frame is None:
            frame = self.frame

        # Convert to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, rejectedCandidates = cv.aruco.detectMarkers(
            gray,
            self.aruco_dict,
            parameters=self.aruco_params
        )

        # If markers are detected
        detected_markers = []
        if len(corners) > 0:
            # Draw detected markers on the frame
            cv.aruco.drawDetectedMarkers(frame, corners, ids)

            # Collect marker details
            for i, marker_id in enumerate(ids):
                # Calculate marker center
                center = np.mean(corners[i][0], axis=0).astype(int)
                detected_markers.append({
                    'id': marker_id[0],
                    'center': center,
                    'corners': corners[i][0]
                })

            # Print marker details
            print("\nDetected Markers:")
            for marker in detected_markers:
                print(f"Marker ID: {marker['id']} - Center: {marker['center']}")

        return detected_markers

    def find_docking_station(self):
        """Find docking station using ArUco markers"""
        # Detect markers
        detected_markers = self.detect_aruco_markers()

        # Look for the specific marker IDs used on docking station: 28, 7, 19, 96
        valid_ids = [28, 7, 19, 96]
        centers = []

        for marker in detected_markers:
            if marker['id'] in valid_ids:
                x, y = marker['center']
                centers.append((x, y))
                print(f"Valid marker found - ID: {marker['id']} at ({x}, {y})")

        if not centers:
            print("No valid docking station markers found")
            return (0, 0)

        # Calculate weighted center
        center_x = sum(x for x, _ in centers) / len(centers)
        center_y = sum(y for _, y in centers) / len(centers)

        center = (int(center_x), int(center_y))
        cv.circle(self.frame, center, 5, (0, 255, 0), 2)
        print(f"Calculated docking station center: {center}")
        
        return center


        

    def calculate_distance(self, marker_center):
        """Estimate distance to docking station based on ArUco marker size"""
        corners, ids, _ = cv.aruco.detectMarkers(
            self.frame, self.aruco_dict, parameters=self.aruco_params
        )
        if ids is None:
            return float('inf')
            
        # ArUco marker physical size (mm)
        MARKER_SIZE = 150 
        
        for i, id in enumerate(ids):
            if id[0] in [28, 7, 19, 96]:
                marker_corners = corners[i][0]
                # Calculate marker pixel size
                pixel_size = np.linalg.norm(marker_corners[0] - marker_corners[1])
               
                # Focal length would need to be calibrated for accurate measurements
                distance = (MARKER_SIZE * 10) / pixel_size
                return distance / 100
        return float('inf')

    def regulate_position(self, displacement_y, displacement_z):
        """Control ROV position relative to docking station"""
        # Threshold increased for more stable approach
        if displacement_y > 3:
            return [0, -displacement_y, 0, 0, 0, 0, 0, 0]  # Move left
        elif displacement_y < -3:
            return [0, displacement_y, 0, 0, 0, 0, 0, 0]  # Move right
        elif displacement_z > 3:
            return [0, 0, -displacement_z, 0, 0, 0, 0, 0]  # Move down
        elif displacement_z < -3:
            return [0, 0, displacement_z, 0, 0, 0, 0, 0]  # Move up
        else:
            return [10, 0, 0, 0, 0, 0, 0, 0]  # Move forward slowly

    def find_grouts(self):
        """Detect floor grid pattern for orientation"""
        lower_bound, upper_bound = (0, 0, 0), (100, 100, 100)
        grouts = cv.inRange(self.down_frame, lower_bound, upper_bound)
        grouts_dilated = cv.dilate(grouts, None, iterations=10)
        canny = cv.Canny(grouts_dilated, 100, 200)
        blurred = cv.GaussianBlur(canny, (11, 13), 0)
        grout_contours, _ = cv.findContours(blurred, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        if self.draw_grouts:
            cv.drawContours(self.down_frame, grout_contours, -1, (0, 255, 0), 3)
            
        return grout_contours

    def get_driving_data(self):
        data = self.driving_data.copy()
        self.driving_data = [0, 0, 0, 0, 0, 0, 0, 0]
        return data

    def find_relative_angle(self):
        """Calculate ROV orientation relative to grid"""
        grout_contours = self.find_grouts()
        
        angle_sum = 0
        angle_counter = 0
        
        for c in grout_contours:
            rect = cv.minAreaRect(c)
            _, (width, height), angle = rect
            area = width * height
            
            frame_height, frame_width = self.frame.shape[:2]
            MAX_AREA = (frame_height * frame_width) * 0.30
            MIN_AREA = (frame_height * frame_width) * 0.05
            
            if (area > MAX_AREA) or (area < MIN_AREA):
                continue
            
            if width < height:
                angle = 90 - angle
            else:
                angle = -angle
                
            angle_sum += angle
            angle_counter += 1

            if self.draw_grout_boxes:
                box = cv.boxPoints(rect)
                box = np.intp(box)
                cv.drawContours(self.down_frame, [box], 0, (0, 0, 255), 2)

        if angle_counter == 0:
            return "NO ANGLE"
            
        return angle_sum / angle_counter
            
    def rotation_commands(self):
        """Adjust ROV rotation to align with grid"""
        angle = self.find_relative_angle()
        if angle == "NO ANGLE":
            return
        
        # Added deadzone and dampening
        DEADZONE = 15  # Larger deadzone
        MAX_ROTATION = 10

        if abs(angle) <= DEADZONE:
            self.angle_good = True
            self.driving_data = [0, 0, 0, 0, 0, 0, 0, 0]
            return
        # Calculate rotation speed based on angle
        rotation_speed = min(abs(angle) * 0.3, MAX_ROTATION)  # Dampening factor of 0.3
    
        # Increased threshold for more stable rotation
        if angle > DEADZONE:
            self.driving_data = [0, 0, 0, rotation_speed, 0, 0, 0, 0]  # Rotate right
            return
        elif angle < -DEADZONE:
            self.driving_data = [0, 0, 0, - rotation_speed, 0, 0, 0, 0]  # Rotate left
            return
        else:
            self.angle_good = True
            self.driving_data = [0, 0, 0, 0, 0, 0, 0, 0]

            return
    def check_precision_docking(self, marker_center):
        """Verify precise alignment for power puck connection"""
        # Tighter tolerances for final docking
        frame_height, frame_width = self.frame.shape[:2]
        PRECISION_THRESHOLD = 0.02  # 2% of frame dimension
        
        x_tolerance = frame_width * PRECISION_THRESHOLD
        y_tolerance = frame_height * PRECISION_THRESHOLD
        
        return (abs(marker_center[0] - frame_width/2) < x_tolerance and 
                abs(marker_center[1] - frame_height/2) < y_tolerance)   
    
  

if __name__ == "__main__":
    # Initialize the AutonomousDocking class
    a = AutonomousDocking()
    
    # Use camera index 1 since that's the one that worked in the test
    cap = cv.VideoCapture(0)
    
    # Set camera resolution (optional)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Failed to open camera")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv.imshow('Raw Frame', frame)
        
        try:
            # Process the frame
            processed_frame, down_frame, data = a.run(frame, frame.copy())
            
            # Display the results
            cv.imshow('Processed Frame', processed_frame)
            cv.imshow('Down Frame', down_frame)
            # Print driving data (optional, for debugging)
            print("Driving data:", data)
        
            # Break the loop if 'q' is pressed
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"Error processing frame: {e}")
            break
    
    # Clean up
    cap.release()
    cv.destroyAllWindows()