import pickle
import gzip

detections_path = "/home/kev/datasets/Replica/room0/gsa_detections_none/frame000000.pkl.gz"
with gzip.open(detections_path, "rb") as f:
    gobs = pickle.load(f)
    
print(gobs.keys())
print(gobs['xyxy'])
