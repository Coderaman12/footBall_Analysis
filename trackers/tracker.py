from ultralytics import YOLO
import supervision as sv
import pickle
import os

class Tracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        
    def detect_frames(self,frames):
        detections = []
        batch_size=20
        
        for i in range(0,len(frames),batch_size):
            detection_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detection_batch 
        return detections
    
    
    def get_object_tracker(self,frames,read_from_stub=False,stub_path=None):
        
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                return pickle.load(f)
            
            return tracks
        
        detections = self.detect_frames(frames)
        
        tracks = {
            "players":[],
            "referees":[],
            "ball":[]
        }
        
        
        
        for frame_num,detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}
            
            # print(cls_names)
            
            # convert to supervision format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            # convert goalkeeper to a player
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]
            
            
            # track object
            detection_with_track = self.tracker.update_with_detections(detection_supervision)
            
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            for frame_detection in detection_with_track:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                
                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}     
                    
                if cls_id == cls_names_inv["player"]:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}     
                    
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                
                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox":bbox} 
                    
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)
            
        return tracks