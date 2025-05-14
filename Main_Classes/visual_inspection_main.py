import cv2 as cv
import numpy as np
import math

class VisualInspection:
    """Visual inspection system using computer vision and ArUco markers. Performs valve intervention."""
    def __init__(self): 
        self.frame = None
        self.driving_data = [0, 0, 0, 0, 0, 0, 0, 0]
        self.structure_detected = False  
        self.valves_found = False
        self.valve_count = 0

        self.lower_hsv = np.array([4, 150, 150])
        self.upper_hsv = np.array([24, 255, 255])

        self.aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_params = cv.aruco.DetectorParameters()
        self.detected_markers_ids = []

        self.tracking_state = "FIND_STRUCTURE" # States: FIND_STRUCTURE, GO_AROUND_STRUCTURE, COMPLETE
        self.structure_center = None
        self.structure_direction = None

    def run(self, front_frame, down_frame=None, left_frame=None, right_frame=None):
        """Run one cycle of the visual inspection system."""
        self.frame = front_frame
        self.down_frame = down_frame if down_frame is not None else np.zeros_like(front_frame)
        self.left_frame = left_frame if left_frame is not None else np.zeros_like(front_frame)
        self.right_frame = right_frame if right_frame is not None else np.zeros_like(front_frame)
        self.update()
        data = self.get_driving_data()
        return self.frame, self.down_frame, self.left_frame, self.right_frame, data, self.detected_markers_ids

    def update(self):
        """Manage process states."""
        if self.tracking_state == "FIND_STRUCTURE":
            self.approach_structure()
        elif self.tracking_state == "GO_AROUND_STRUCTURE":
            self.follow_structure_and_search()
        elif self.tracking_state == "COMPLETE":
            self.driving_data = [0, 0, 0, 0, 0, 0, 0, 0]

    def approach_structure(self):
        """Find and approach the structure."""
        self.detect_structure()
        if self.structure_detected:
            print("Structure detected! Switching to GO_AROUND_STRUCTURE.")
            self.tracking_state = "GO_AROUND_STRUCTURE"
        else:
            self.driving_data = [5, 0, 0, 2, 0, 0, 0, 0]

    def follow_structure_and_search(self):
        """Move around the structure and search for markers and valves."""
        self.detect_structure()
        if not self.structure_detected:
            print("Lost structure. Returning to FIND_STRUCTURE.")
            self.tracking_state = "FIND_STRUCTURE"
            return

        self.regulate_position()
        self.detect_markers()
        self.find_valve()

        if self.valves_found and 5 <= len(self.detected_markers_ids) <= 10:
            print("Valves and required number of markers found! Task complete.")
            self.tracking_state = "COMPLETE"

    def detect_structure(self):
        """Detect a subsea structure."""
        hsv = cv.cvtColor(self.frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, self.lower_hsv, self.upper_hsv)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        valid_contours = [cnt for cnt in contours if cv.contourArea(cnt) > 300]

        if valid_contours:
            largest = max(valid_contours, key=cv.contourArea)
            cv.drawContours(self.frame, [largest], -1, (0, 255, 0), 2)

            M = cv.moments(largest)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                self.structure_center = (cX, cY)
                cv.circle(self.frame, self.structure_center, 5, (255, 0, 0), -1)

            rect = cv.minAreaRect(largest)
            self.structure_direction = rect[2]

            self.structure_detected = True
        else:
            self.structure_detected = False
            self.structure_center = None
            self.structure_direction = None

    def regulate_position(self):
        """Control ROV position relative to structure."""
        if self.structure_center is None:
            return

        frame_height, frame_width = self.frame.shape[:2]
        frame_center = (frame_width // 2, frame_height // 2)

        commands = self.get_navigation_command(self.structure_center, frame_center, 20)
        print("Navigation Commands:", commands)

    def detect_markers(self):
        """Detect ArUco markers in the current frame and update the markers list."""
        gray = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        corners, ids, _ = cv.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None:
            cv.aruco.drawDetectedMarkers(self.frame, corners, ids)
            for i, marker_id in enumerate(ids):
                id_int = int(marker_id[0])
                if 1 <= id_int <= 99 and id_int not in self.detected_markers_ids:
                    self.detected_markers_ids.append(id_int)
                    print(f"New marker detected! ID: {id_int}")

                top_left = tuple(corners[i][0][0].astype(int))
                cv.putText(self.frame, f"ID: {id_int}",
                           (top_left[0] + 10, top_left[1] - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv.circle(self.frame, top_left, 5, (0, 255, 0), -1)

    def get_driving_data(self):
        """Return driving data."""
        data = self.driving_data.copy()
        if self.frame is not None:
            cmd_text = f"Drive: [{data[0]:.1f}, {data[1]:.1f}, {data[2]:.1f}, {data[3]:.1f}]"
            cv.putText(self.frame, cmd_text, (10, self.frame.shape[0] - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return data
    
    def find_valve(self):
        """Detect valves."""
        lower_hsv = np.array([0, 0, 200])
        upper_hsv = np.array([180, 30, 255])
        hsv = cv.cvtColor(self.frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower_hsv, upper_hsv)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        valves_detected = 0
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area < 300:  # Filter out small noise
                continue

            rect = cv.minAreaRect(cnt)
            (center_x, center_y), (width, height), angle = rect

            aspect_ratio = max(width, height) / min(width, height) if min(width, height) != 0 else 0

            if 1.2 < aspect_ratio < 1.5 and 500 < area < 5000:
                box = np.intp(cv.boxPoints(rect))
                cv.drawContours(self.frame, [box], -1, (0, 0, 255), 2)
                valves_detected += 1

        if valves_detected == 2:
            self.valves_found = True

    def calculate_displacement(self, target_center, image_center):
        """Calculate displacement between target and image center."""
        dx = target_center[0] - image_center[0]
        dy = target_center[1] - image_center[1]
        return dx, dy

    def determine_commands(self, dx, dy, threshold):
        """Determine navigation commands based on displacement."""
        commands = []
        if abs(dx) > threshold:
            commands.append("RIGHT" if dx > 0 else "LEFT")
        if abs(dy) > threshold:
            commands.append("DOWN" if dy > 0 else "UP")
        return commands

    def get_navigation_command(self, target_center, image_center, threshold=10):
        """Get navigation commands based on target position."""
        if not target_center:
            return ["SEARCH"]
        dx, dy = self.calculate_displacement(target_center, image_center)
        commands = self.determine_commands(dx, dy, threshold)
        if not commands:
            commands.append("STOP")
        return commands

    def perform_valve_operation(self):
        """Perform valve intervention."""
        pass

if __name__ == "__main__":
    inspection = VisualInspection()
    cap = cv.VideoCapture(".../camerafeed/videos/visual.mp4")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to read frame")
            break
        # Process the frame
        processed_frame, down_frame, left_frame, right_frame, data, markers = inspection.run(frame)
        
        # Display the processed frame
        cv.imshow("Visual Inspection", processed_frame)
        
        # Break the loop if 'q' is pressed
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
