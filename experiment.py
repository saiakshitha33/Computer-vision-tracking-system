import torch
import numpy as np
import cv2
from ultralytics import YOLO
from pathlib import Path
import time
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
import supervision as sv
from bytetrack.byte_tracker import BYTETracker
from strongsort.strong_sort import StrongSORT
from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.tracker import Tracker
from util import get_config
from supervision import ColorPalette

SAVE_VIDEO = True
TRACKER = "strongsort"  # Options: "bytetrack", "strongsort", "deepsort"

class ObjectDetection:

    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        # Load the model
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
        self.box_annotator = sv.BoundingBoxAnnotator()
        self.frame_rate = 30
        self.nth_frame = 5
        self.real_world_distance = 1.2
        self.trajectories = {}  # Dictionary to store trajectories for each ID
        self.velocities = {}  # Dictionary to store velocities for each ID
        self.accelerations = {}  # Dictionary to store accelerations for each ID
        self.id_start_times = {}  # Dictionary to store start times for each ID
        self.id_durations = {}  # Dictionary to store total durations for each ID
        self.roi_id_start_times = {}  # Dictionary to store start times for each ID in ROI
        self.roi_id_durations = {}  # Dictionary to store total durations for each ID in ROI
        self.roi = None  # Set later in the __call__ method
        self.feature_window = 30  # Number of frames to consider for feature extraction
        self.ml_model = None  # We'll initialize this later

        # Load the appropriate tracker based on the selection
        reid_weights = Path("weights/poultrymodel_feature_extractor_weights.pt")

        if TRACKER == "bytetrack":
            tracker_config = "bytetrack/configs/bytetrack.yaml"
            cfg = get_config()
            cfg.merge_from_file(tracker_config)
            self.tracker = BYTETracker(
                track_thresh=cfg.bytetrack.track_thresh,
                match_thresh=cfg.bytetrack.match_thresh,
                track_buffer=cfg.bytetrack.track_buffer,
                frame_rate=cfg.bytetrack.frame_rate
            )
        elif TRACKER == "strongsort":
            tracker_config = "strongsort/configs/strongsort.yaml"
            cfg = get_config()
            cfg.merge_from_file(tracker_config)
            self.tracker = StrongSORT(
                reid_weights,
                torch.device(self.device),
                False,
                max_dist=cfg.strongsort.max_dist,
                max_iou_dist=cfg.strongsort.max_iou_dist,
                max_age=cfg.strongsort.max_age,
                max_unmatched_preds=cfg.strongsort.max_unmatched_preds,
                n_init=cfg.strongsort.n_init,
                nn_budget=cfg.strongsort.nn_budget,
                mc_lambda=cfg.strongsort.mc_lambda,
                ema_alpha=cfg.strongsort.ema_alpha,
            )
        elif TRACKER == "deepsort":
            self.tracker = DeepSort(max_age=5)

    # Load YOLOv8 model
    def load_model(self):
        model = YOLO("best-6.pt")  # load a pretrained YOLOv8 model
        model.fuse()
        return model

    # Feature extraction from trajectory data
    def extract_features(self, trajectory):
        if len(trajectory) < self.feature_window:
            return None
        
        recent_trajectory = trajectory[-self.feature_window:]
        x = [p[0] for p in recent_trajectory]
        y = [p[1] for p in recent_trajectory]
        
        # Smooth the trajectory using Savitzky-Golay filter
        x_smooth = savgol_filter(x, window_length=7, polyorder=3)
        y_smooth = savgol_filter(y, window_length=7, polyorder=3)
        
        # Calculate velocities and accelerations
        vx = np.diff(x_smooth)
        vy = np.diff(y_smooth)
        ax = np.diff(vx)
        ay = np.diff(vy)
        
        # Calculate curvature
        curvature = np.abs(vx[:-1] * ay - vy[:-1] * ax) / (vx[:-1]**2 + vy[:-1]**2)**1.5
        
        # Extract statistical features
        features = [
            np.mean(vx), np.std(vx), np.max(np.abs(vx)),
            np.mean(vy), np.std(vy), np.max(np.abs(vy)),
            np.mean(ax), np.std(ax), np.max(np.abs(ax)),
            np.mean(ay), np.std(ay), np.max(np.abs(ay)),
            np.mean(curvature), np.std(curvature), np.max(curvature)
        ]
        return features

    # Prepare the training data
    def prepare_training_data(self):
        X = []
        y = []
        trajectory_ids = list(self.trajectories.keys())
        
        for i in range(len(trajectory_ids)):
            for j in range(i + 1, len(trajectory_ids)):
                id1, id2 = trajectory_ids[i], trajectory_ids[j]
                features1 = self.extract_features(self.trajectories[id1])
                features2 = self.extract_features(self.trajectories[id2])
                
                if features1 is not None and features2 is not None:
                    X.append(features1 + features2)
                    # Assume different IDs belong to different objects
                    y.append(0 if id1 != id2 else 1)
        return np.array(X), np.array(y)

    # Train the Random Forest model
    def train_model(self):
        X, y = self.prepare_training_data()
        if len(X) > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.ml_model.fit(X_train, y_train)
            print(f"Model accuracy: {self.ml_model.score(X_test, y_test)}")
        else:
            print("Not enough data to train the model")

    # Predict whether two IDs belong to the same object
    def predict_same_object(self, id1, id2):
        if self.ml_model is None:
            return False
        
        features1 = self.extract_features(self.trajectories[id1])
        features2 = self.extract_features(self.trajectories[id2])
        
        if features1 is None or features2 is None:
            return False
        
        combined_features = features1 + features2
        prediction = self.ml_model.predict([combined_features])[0]
        return prediction == 1

    # Update process_track to include frame numbers
    def process_track(self, frame, bbox, tracked_id, frame_width, frame_count, current_time):
        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)
        center = (center_x, center_y, frame_count)  # Include frame number

        if tracked_id not in self.trajectories:
            self.trajectories[tracked_id] = []
        self.trajectories[tracked_id].append(center)

    # Main tracking loop with ID switch correction
    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        if SAVE_VIDEO:
            outputvid = cv2.VideoWriter('result_tracking.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20, (1280, 720))

        tracker = self.tracker

        frame_count = 0
        train_interval = 100  # Train the model every 100 frames
        color_palette = ColorPalette(256)
        id_colors = {}

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = self.predict(frame)
                frame_count += 1

                # Train the model periodically
                if frame_count % train_interval == 0:
                    self.train_model()

                # Perform ID switch correction
                if self.ml_model is not None:
                    corrected_ids = {}
                    for id1 in list(self.trajectories.keys()):
                        for id2 in list(self.trajectories.keys()):
                            if id1 != id2 and self.predict_same_object(id1, id2):
                                new_id = min(id1, id2)
                                if new_id not in corrected_ids:
                                    corrected_ids[new_id] = []
                                corrected_ids[new_id].extend(self.trajectories[id1])
                                corrected_ids[new_id].extend(self.trajectories[id2])
                                if id1 in self.trajectories:
                                    del self.trajectories[id1]
                                if id2 in self.trajectories:
                                    del self.trajectories[id2]
                    for new_id, merged_trajectory in corrected_ids.items():
                        self.trajectories[new_id] = merged_trajectory

                for result in results:
                    outputs = tracker.update(result, frame)
                    for output in outputs:
                        bbox = output[:4]
                        tracked_id = output[4]
                        self.process_track(frame, bbox, tracked_id, frame.shape[1], frame_count, time.time())

                        # Assign colors to track IDs
                        if tracked_id not in id_colors:
                            id_colors[tracked_id] = color_palette(tracked_id)

                        # Draw bounding box and trajectory
                        color = id_colors[tracked_id]
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)

                        if len(self.trajectories[tracked_id]) > 1:
                            for i in range(1, len(self.trajectories[tracked_id])):
                                point1 = self.trajectories[tracked_id][i-1][:2]
                                point2 = self.trajectories[tracked_id][i][:2]
                                cv2.line(frame, point1, point2, color, 2)

                if SAVE_VIDEO:
                    outputvid.write(frame)

                cv2.imshow('Tracking', frame)
                if cv2.waitKey(5) & 0xFF == 27:
                    break

        finally:
            if SAVE_VIDEO:
                outputvid.release()
            cap.release()
            cv2.destroyAllWindows()

detector = ObjectDetection(capture_index="sample_video.mp4")
detector()
