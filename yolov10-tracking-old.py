import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
import os
import yaml
from easydict import EasyDict as edict
from pathlib import Path

import supervision as sv
from bytetrack.byte_tracker import BYTETracker
from strongsort.strong_sort import StrongSORT



SAVE_VIDEO = True
# "bytetrack" "strongsort"
TRACKER  =  "bytetrack"

from util import get_config
from supervision import ColorPalette


class ObjectDetection:

    def __init__(self, capture_index):
       
        self.capture_index = capture_index
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        
        self.model = self.load_model()
        
        self.CLASS_NAMES_DICT = self.model.model.names
    
        self.box_annotator = sv.BoundingBoxAnnotator()

        self.frame_rate = 30
        self.nth_frame = 5
        self.real_world_distance = 1.2
        self.trajectories = {}  # Dictionary to store trajectories for each ID
        self.velocities = {}  # Dictionary to store velocities for each ID
        self.accelerations = {}  # Dictionary to store accelerations for each ID
        
        # no reid weights for bytetrackers :) but using for strongsort
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
        elif TRACKER == "strongsort" :
            tracker_config = "strongsort/configs/strongsort.yaml"
            cfg = get_config()
            cfg.merge_from_file(tracker_config)
    
            self.tracker = StrongSORT (
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
       


    def load_model(self):
       
        model = YOLO("best-6.pt")  # load a pretrained YOLOv8n model
        model.fuse()
    
        return model


    def predict(self, frame):
       
        results = self.model(frame)
        
        return results

    def calculate_distance_in_meters(self, point1, point2, frame_width):
        # Calculate the Euclidean distance between two points in pixels
        pixel_distance = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
        # Convert pixel distance to meters
        meters_per_pixel = self.real_world_distance / frame_width
        return pixel_distance * meters_per_pixel
    

    def draw_results(self, frame, results):
        xyxys = []
        confidences = []
        class_ids = []
        detections = []
        boxes = []
        for result in results:
            # return a list of class ids
            class_id = result.boxes.cls.cpu().numpy().astype(int) 

            if len(class_id) == 0:
                continue

            if len(class_id) > 1:
                class_id = class_id[0]
            
            xyxys.append(result.boxes.xyxy.cpu().numpy())
            confidences.append(result.boxes.conf.cpu().numpy())
            class_ids.append(result.boxes.cls.cpu().numpy().astype(int))
            boxes.append(result.boxes)
            # Setup detections for visualization
            detections = sv.Detections(
                        xyxy=result.boxes.xyxy.cpu().numpy(),
                        confidence=result.boxes.conf.cpu().numpy(),
                        class_id=result.boxes.cls.cpu().numpy().astype(int),
                        )

            # Format custom labels
            self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                           
            for xyxy, mask, confidence, class_id, tracker_id, data
            in detections]
        
        # Annotate and display frame
        frame = self.box_annotator.annotate(scene=frame, detections=detections)
        return frame, boxes
       

    def __call__(self):

        cap = cv2.VideoCapture(self.capture_index)

        if SAVE_VIDEO:
            outputvid = cv2.VideoWriter('result_tracking.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20, (1280,720))
        # setup tracker
        tracker = self.tracker

        # if tracker is using model then warmup
        if hasattr(tracker, 'model'):
            if hasattr(tracker.model, 'warmup'):
                tracker.model.warmup()

        outputs = [None]
        curr_frames, prev_frames = None, None

        highest_track_id = 0
        prev_id_swiches = 0
        gt_objects = 19
        id_switches = 0
        number_of_detections = 0
        false_negatives = 0
        false_positives = 0

        id_switches_list = []
        false_positives_list = []
        false_negatives_list = []

        frame_count = 0

        while cap.isOpened():
            start_time = time()
            ret, frame = cap.read()
            if ret is False:
              break
            results = self.predict(frame)
            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)
            if results is None:
              break
            frame, _ = self.draw_results(frame, results)

            frame_count += 1
            
            if hasattr(tracker, 'tracker') and hasattr(tracker.tracker, 'camera_update'):
                if prev_frames is not None and curr_frames is not None:  # camera motion compensation
                    tracker.tracker.camera_update(prev_frames, curr_frames)

            number_of_detections = 0
            frame_width = frame.shape[1]  # Get the width of the frame for convers

            for result in results:
                outputs[0] = tracker.update(result,frame)
        
                for j, (output) in enumerate(outputs[0]):
                    bbox = output[0:4]
                    tracked_id = output[4]
                    number_of_detections += 1
                    if tracked_id > highest_track_id:
                      highest_track_id = tracked_id


                     # Calculate the center of the bounding box
                    center_x = int((bbox[0] + bbox[2]) / 2)
                    center_y = int((bbox[1] + bbox[3]) / 2)
                    center = (center_x, center_y)

                    if tracked_id not in self.trajectories:
                        self.trajectories[tracked_id] = []
                    self.trajectories[tracked_id].append(center)


                    # Keep only the last 100 points
                    if len(self.trajectories[tracked_id]) > 100:
                        self.trajectories[tracked_id] = self.trajectories[tracked_id][-100:]

                    velocity = 0
                    acceleration = 0

                    # Calculate velocity and acceleration every nth frame
                    if frame_count % self.nth_frame == 0 and len(self.trajectories[tracked_id]) > 1:
                        last_point = self.trajectories[tracked_id][-1]
                        prev_point = self.trajectories[tracked_id][-2]
                        distance_meters = self.calculate_distance_in_meters(last_point, prev_point, frame_width)
                        velocity = distance_meters * self.frame_rate  # Convert to meters per second

                        if tracked_id not in self.velocities:
                            self.velocities[tracked_id] = []
                        self.velocities[tracked_id].append(velocity)

                        if len(self.velocities[tracked_id]) > 1:
                            prev_velocity = self.velocities[tracked_id][-2]
                            acceleration = (velocity - prev_velocity) * self.frame_rate  # Convert to meters per second squared

                            if tracked_id not in self.accelerations:
                                self.accelerations[tracked_id] = []
                            self.accelerations[tracked_id].append(acceleration)

                    top_left = (int(bbox[0]), int(bbox[1]))
                    cv2.putText(frame, f"ID: {tracked_id}", (top_left[0], top_left[1] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, f"Vel: {velocity:.2f} m/s", (top_left[0], top_left[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, f"Acc: {acceleration:.2f} m/s^2", (top_left[0], top_left[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # cls = output[5]
                    # conf = output[6]
                 


            id_switches = highest_track_id-gt_objects

            id_switches -= prev_id_swiches

            if id_switches < 0:
              id_switches = 0
            
            highest_track_id = 0
            prev_id_swiches = id_switches

            print(number_of_detections)

            false_negatives = gt_objects - number_of_detections
            if false_negatives < 0:
              false_negatives = 0

            false_positives = number_of_detections - gt_objects
            if false_positives < 0:
              false_positives = 0

            print(id_switches)
            print(false_negatives)
            print(false_positives)


             # tracker acc for current frame
            mota = abs(1 - ((false_negatives+false_positives+id_switches) / gt_objects))
            print("Mota frame: ", mota)


            id_switches_list.append(id_switches)
            false_positives_list.append(false_positives)
            false_negatives_list.append(false_negatives)

            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            #cv2.imshow('YOLOv8 Detection', frame)

            for tracked_id, trajectory in self.trajectories.items():
                for point in trajectory:
                    cv2.circle(frame, point, 2, (0, 255, 0), -1)


            if SAVE_VIDEO:
                frame = cv2.resize(frame, (1280,720))
                outputvid.write(frame)
                print("frame saved")
            if cv2.waitKey(5) & 0xFF == 27:
                break

        # calculate overall tracker acc 
        mean_id_switches = np.mean(np.array(id_switches_list))
        mean_fp = np.mean(np.array(false_positives_list))
        mean_fn = np.mean(np.array(false_negatives_list))

        # tracker acc for current frame
        mota_mean = 1 - ((false_negatives+false_positives+id_switches) / gt_objects)
        print("Mota overall: ", mota_mean)

        if SAVE_VIDEO:
            outputvid.release()
        cap.release()
        cv2.destroyAllWindows()
        
        
    
detector = ObjectDetection(capture_index="sample_video.mp4")
detector()