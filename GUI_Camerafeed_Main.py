import threading
import time
import numpy as np
from Main_Classes.PipelineInspection import PipelineInspection
from Main_Classes.autonomous_docking_main import AutonomousDocking
from Main_Classes.Coralheadmodeling import MVS
from Main_Classes.visual_inspection_main import VisualInspection
import cv2
import multiprocessing as mp
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import datetime
import random
import os
import shutil


X_AXIS = 1
Y_AKSE = 0
Z_AKSE = 6
R_AKSE = 2

GST_FEED_STEREO_L = "-v udpsrc multicast-group=224.1.1.1 auto-multicast=true port=5000 ! application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96 ! rtph264depay ! h264parse ! decodebin ! videoconvert ! appsink sync=false"
GST_FEED_STEREO_R = "-v udpsrc multicast-group=224.1.1.1 auto-multicast=true port=5001 ! application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96 ! rtph264depay ! h264parse ! decodebin ! videoconvert ! appsink sync=false"
GST_FEED_DOWN = "-v udpsrc multicast-group=224.1.1.1 auto-multicast=true port=5002 ! application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96 ! rtph264depay ! h264parse ! decodebin ! videoconvert ! appsink sync=false"
GST_FEED_MANIPULATOR = "-v udpsrc multicast-group=224.1.1.1 auto-multicast=true port=5003 ! application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96 ! rtph264depay ! h264parse ! decodebin ! videoconvert ! appsink sync=false"


class Camera:
    def __init__(self, name, gst_feed=None):
        self.name = name  # Name of camera
        self.gst = gst_feed  # Gstreamer feed

    def get_frame(self):
        ret, frame = self.cam.read()
        if not ret:
            print("Error reading frame")
            return
        return frame

    def open_cam(self):
        if self.gst is None:
            self.cam = cv2.VideoCapture(0)
            if not self.isOpened:
                print(f"Error opening camera {self.name}")
                return False
            print(f"{self.name} Camera opened")
            return True
        else:
            self.cam = cv2.VideoCapture(self.gst, cv2.CAP_GSTREAMER)
            if not self.isOpened:
                print("Error opening camera")
                return False
            print(f"{self.name} Camera opened")
            return True

    def release_cam(self):
        self.cam.release()

    @property
    def isOpened(self):
        return self.cam.isOpened()


class CameraManager:
    def __init__(self) -> None:
        self.frame_manipulator = None
        self.frame_stereoR = None
        self.frame_down = None
        self.frame_stereoL = None
        self.frame_manual = None
        self.frame_test = None

        self.cam_stereoL = None
        self.cam_stereoR = None
        self.cam_down = None
        self.cam_manipulator = None
        self.cam_manual = None
        self.cam_test = None

        self.recording = False

        self.active_cameras = []

    def get_frame_stereo_L(self):
        self.frame_stereoL = self.cam_stereoL.get_frame()
        return self.frame_stereoL

    def get_frame_stereo_R(self):
        self.frame_stereoR = self.cam_stereoR.get_frame()
        return self.frame_stereoR

    def get_frame_down(self):
        self.frame_down = self.cam_down.get_frame()
        return self.frame_down

    def get_frame_manipulator(self):
        self.frame_manipulator = self.cam_manipulator.get_frame()
        return self.frame_manipulator

    def get_frame_manual(self):
        self.frame_manual = self.cam_manual.get_frame()
        return self.frame_manual

    def get_frame_test(self):
        self.frame_test = self.cam_test.get_frame()
        return self.frame_test

    def get_frame_from_cam(self, cam: Camera):
        frame = cam.get_frame()
        return frame

    def start_stereo_cam_L(self):
        self.cam_stereoL = Camera("StereoL", GST_FEED_STEREO_L)
        print("Starting camera: StereoL")
        success = self.cam_stereoL.open_cam()
        if success:
            self.active_cameras.append(self.cam_stereoL)

    def start_stereo_cam_R(self):
        self.cam_stereoR = Camera("StereoR", GST_FEED_STEREO_R)
        print("Starting camera: StereoR")
        success = self.cam_stereoR.open_cam()
        if success:
            self.active_cameras.append(self.cam_stereoR)

    def start_down_cam(self):
        self.cam_down = Camera("Down", GST_FEED_DOWN)
        print("Starting camera: Down")
        success = self.cam_down.open_cam()
        if success:
            self.active_cameras.append(self.cam_down)

    def start_manipulator_cam(self):
        self.cam_manipulator = Camera("Manipulator", GST_FEED_MANIPULATOR)
        print("Starting camera: Manipulator")
        success = self.cam_manipulator.open_cam()
        if success:
            self.active_cameras.append(self.cam_manipulator)

    def start_manual_cam(self):
        self.cam_manual = Camera("Manual")
        print("Starting camera: Manual")
        success = self.cam_manual.open_cam()
        if success:
            self.active_cameras.append(self.cam_manual)

    def start_test_cam(self):
        self.cam_test = Camera("Test")
        print("Starting camera: Test")
        success = self.cam_test.open_cam()
        if success:
            self.active_cameras.append(self.cam_test)

    def start(self):
        # self.start_down_cam()
        # self.start_stereo_cam_L()
        # self.start_stereo_cam_R()
        # self.start_manipulator_cam()
        # self.start_manual_cam()
        pass

    def close_all(self):
        if self.cam_down is not None and self.cam_down.isOpened:
            self.cam_down.release_cam()

        if self.cam_stereoL is not None and self.cam_stereoL.isOpened:
            self.cam_stereoL.release_cam()

        if self.cam_stereoR is not None and self.cam_stereoR.isOpened:
            self.cam_stereoR.release_cam()

        if self.cam_manipulator is not None and self.cam_manipulator.isOpened:
            self.cam_manipulator.release_cam()

        if self.cam_manual is not None and self.cam_manual.isOpened:
            self.cam_manual.release_cam()

        if self.cam_test is not None and self.cam_test.isOpened:
            self.cam_test.release_cam()

    def setup_video(self, name):
        self.videoresult = cv2.VideoWriter(
            f"camerafeed/output/{name}.avi",
            cv2.VideoWriter_fourcc(*"MJPG"),
            10,
            (
                int(self.active_cameras[0].cam.get(3)),
                int(self.active_cameras[0].cam.get(4)),
            ),
        )

    # Run this to start recording, and do a keyboard interrupt (ctrl + c) to stop recording

    # TODO make it so that it can record from multiple cameras at the same time
    # TODO MAKE IT WORK (Only reord if there is a camera selected) <<<<<<<<<<
    def record_video(self):
        # if self.recording == False:
        #     if len(self.active_cameras) == 0:
        #         print("No camera selected")

        # else:
        #     # Makes a new videofile if it is not already recording (Default)
        if self.recording == False:
            self.setup_video(
                f"Video{self.active_cameras[0].name}{datetime.datetime.now()}"
            )
            self.recording = True

        # While recording, it will write the frames to the videofile
        if self.recording == True:
            cur_cam = self.active_cameras[0]

            frame = self.get_frame_from_cam(cur_cam)
            self.videoresult.write(frame)
            return frame
        else:
            self.videoresult.release()

    def save_image(self):
        if len(self.active_cameras) == 0:
            print("No camera selected")
        else:
            print("Cameras detected, amount: ", len(self.active_cameras))
            for i in range(len(self.active_cameras)):
                cur_cam = self.active_cameras[i]
                frame_to_save = self.get_frame_from_cam(cur_cam)
                cv2.imwrite(
                    f"camerafeed/output/Img_Cam_{cur_cam.name}{datetime.datetime.now()}.jpg",
                    frame_to_save,
                )
            print("Image(s) saved")


class ExecutionClass:
    def __init__(self, driving_queue, manual_flag):

        camera_matrix = np.array([[942.6, 0, 998.5], [0, 940.4, 483.6], [0, 0, 1]])
        dist_coeffs = np.array([[-0.400, 0.210, 7.31e-3, -6.25e-3, -7.03e-2]]).reshape(-1, 1)


        self.PipelineInspection = PipelineInspection()
        self.Docking = AutonomousDocking(camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        self.VisualInspection = VisualInspection()
        self.Camera = CameraManager()
        self.tD_Reconstruction =  MVS(
            image_dir="3Dimages",  # Temporary directory to store captured images
            output_dir="PointCloud_output",
            scale_factor=2.0,
            feature_type='SIFT',
            enable_bundle_adjustment=True
        )
        os.makedirs("3Dimages", exist_ok=True)
        os.makedirs("PointCloud_output", exist_ok=True)
    
        self.counter = 0
        self.done = False
        self.manual_flag = manual_flag
        self.driving_queue = driving_queue

    def update_down(self):
        self.frame_down = self.Camera.get_frame_down()

    def update_stereo_L(self):
        self.frame_stereoL = self.Camera.get_frame_stereo_L()

    def update_stereo_R(self):
        self.frame_stereoR = self.Camera.get_frame_stereo_R()

    def update_manipulator(self):
        self.frame_manipulator = self.Camera.get_frame_manipulator()

    def update_manual(self):
        self.frame_manual = self.Camera.get_frame_manual()

    def update_test_cam(self):
        self.frame_test = self.Camera.get_frame_test()

    def show(self, frame, name="frame"):
        cv2.imshow(name, frame)
        if cv2.waitKey(1) == ord("q"):
            self.stop_everything()

    def testing_for_torr(self):
        self.done = False
        while not self.done:
            self.update_stereo_L()
            self.update_stereo_R()
            self.show(self.frame_stereoL, "StereoL")
            self.show(self.frame_stereoR, "StereoR")
            QApplication.processEvents()

    def camera_test(self):
        while self.done:
            # self.update_manual()
            self.update_down()
            self.update_stereo_L()
            # self.update_stereo_R()
            # self.update_manip()

            self.show(self.frame_down, "Down")
            # self.show(self.frame_manual, "Manual")
            self.show(self.frame_stereoL, "StereoL")
            # self.show(self.frame_stereoR, "StereoR")
            # self.show(self.frame_manipulator, "Manip")
            QApplication.processEvents()

    def send_data_test(self):
        self.done = False
        start = 0
        while not self.done:
            cur_time = time.time()
            if (cur_time - start) > 0.02:
                random_data = [random.randint(0, 10) for _ in range(8)]
                self.send_data_to_rov(random_data)

                QApplication.processEvents()
                start = time.time()

    def sleep_func(self):
        threading.Timer(1000, self.sleep_func).start()

    def pipeline(self):
        self.done = False
        self.Camera.start_manipulator_cam()  # TODO should be down frame
        while not self.done and self.manual_flag.value == 0:
            self.update_manipulator()  # TODO Should be down frame
            pipeline_frame, driving_data_packet, marker_ids, nav_angle, nav_vector = self.PipelineInspection.run(
                self.frame_manipulator
            )
            self.show(pipeline_frame, "Pipeline")
            self.send_data_to_rov(driving_data_packet, 
                                marker_ids=marker_ids,
                                nav_angle=nav_angle,
                                nav_vector=nav_vector)
            QApplication.processEvents()
        else:
            self.stop_everything()

    def inspection(self):
        """Manage the visual inspection process."""
        self.done = False
        self.Camera.start_manipulator_cam()
        self.Camera.start_down_cam()
        self.Camera.start_stereo_cam_L()
        self.Camera.start_stereo_cam_R()

        while not self.done and self.manual_flag.value == 0:
            self.update_manipulator()
            self.update_down()
            self.update_stereo_L()
            self.update_stereo_R()
            inspection_frame, frame_under, frame_left, frame_right, driving_data_packet = self.VisualInspection.run(self.frame_manipulator, self.frame_down, self.frame_stereoL, self.frame_stereoR)
            self.show(inspection_frame, "Visual Inspection")
            self.show(frame_under, "Down")
            self.show(frame_left, "Left")
            self.show(frame_right, "Right")
            self.send_data_to_rov(driving_data_packet)
        else:
            self.stop_everything()

    def reconstruction(self):
        self.done = False
        self.Camera.start_manipulator_cam() 
        image_count = 0
        os.makedirs("3Dimages", exist_ok=True)
        for file in os.listdir("3Dimages"):
            os.remove(os.path.join("3Dimages", file))
        while not self.done and self.manual_flag.value and image_count < 60:
            self.update_manipulator()
            cv2.imwrite(f"3Dimages/image_{image_count:03d}.jpg", self.frame_manipulator)
            image_count += 1
            frame_copy = self.frame_manipulator.copy()
            cv2.putText(frame_copy, f"{image_count}/60", (30, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.show(frame_copy, "Reconstruction")
            time.sleep(0.5) #capture rate 
            if image_count > 10:
                Reconstruction = self.tD_Reconstruction.run(
                    self.frame_manipulator
                )
                self.show(Reconstruction, "Reconstruction")
                self.send_data_to_rov(Reconstruction)
            QApplication.processEvents()
        else: 
            self.stop_everything()

    def docking(self):
        self.done = False
        self.Camera.start_manipulator_cam()  #For marker detection
        self.Camera.start_down_cam()    # for seeing the entire pallet
        while not self.done and self.manual_flag.value == 0:
            # Needs stereo L, and Down Cameras
            self.update_manipulator()
            self.update_down()  
            docking_frame, frame_under, driving_data_packet = self.Docking.run(
                self.frame_manipulator, self.frame_down
            )  # TODO should be down camera
            self.show(docking_frame, "Docking")
            self.show(frame_under, "Frame Under")
            self.send_data_to_rov(driving_data_packet)
            QApplication.processEvents()
        else:
            self.stop_everything()

    def send_data_to_rov(self, datapacket, marker_ids= None, nav_angle = None, nav_vector=None):
        data_to_send = {"autonomdata": datapacket,
                        "marker_ids": marker_ids,
                        "navigation_angle": nav_angle,
                        "navigation_vector": nav_vector}
        self.driving_queue.put((2, data_to_send))
    

    def normal_camera(self):
        self.done = False
        self.Camera.start_manual_cam()
        while not self.done:
            self.update_manual()
            self.show(self.frame_manual, "Manual")
            QApplication.processEvents()

    def show_all_cameras(self):
        self.done = False
        self.Camera.start_stereo_cam_L()
        self.Camera.start_stereo_cam_R()
        self.Camera.start_down_cam()
        self.Camera.start_manipulator_cam()
        while not self.done:
            self.update_stereo_L()
            self.update_stereo_R()
            self.update_down()
            self.update_manipulator()
            self.show(self.frame_stereoL, "StereoL")
            self.show(self.frame_stereoR, "StereoR")
            self.show(self.frame_down, "Down")
            self.show(self.frame_manipulator, "Manip")

    def stop_everything(self):
        print("Stopping other processes, returning to manual control")
        try:
            self.done = True
            cv2.destroyAllWindows()
            self.Camera.close_all()
            self.Camera.active_cameras = []
        except:
            pass

    def pipeline_test(self):
        print("Running Pipeline!")

    def record(self):
        self.done = False

        # If the amera is already recording, then this stops it
        if self.Camera.recording:
            self.Camera.recording = False
            print("Recording stopped")
            self.done = True

        active_cams = self.Camera.active_cameras
        if len(active_cams) >= 1:
            print("Cameras deteced, amount: ", len(active_cams))
        else:
            print("No active cameras")
            self.done = True

        while not self.done and len(self.Camera.active_cameras) >= 1:
            frame = self.Camera.record_video()
            self.show(frame, "Recording")
            QApplication.processEvents()
        else:
            self.Camera.recording = False
            cv2.destroyWindow("Recording")
            print("Recording stopped or no active cameras")
            QApplication.processEvents()

    def save_image(self):
        self.Camera.save_image()


if __name__ == "__main__":
    cam = CameraManager()
    execution = ExecutionClass()
    while True:
        execution.update_stereo()
        # execution.show(execution.frame_down, "Down")
        execution.show(execution.frame_stereoL, "StereoL")
        execution.show(execution.frame_stereoR, "StereoR")
        # execution.show(execution.frame_manipulator, "Manip")
