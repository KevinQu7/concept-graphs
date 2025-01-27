import gzip
from pathlib import Path
import pickle as pkl


path = "/home/kev/datasets/Replica/office3/pcd_saves/full_pcd_none_FULL_overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub.pkl.gz"
with gzip.open(Path(path), "rb") as f:
        loaded_data = pkl.load(f)
        print(loaded_data)