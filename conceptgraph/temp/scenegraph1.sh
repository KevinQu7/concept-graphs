SCENE_NAME=office3
PKL_FILENAME=full_pcd_none_FULL_overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub_post.pkl.gz  # Change this to the actual output file name of the pkl.gz file

python scenegraph/build_scenegraph_cfslam.py \
    --mode extract-node-captions \
    --cachedir ${REPLICA_ROOT}/${SCENE_NAME}/sg_cache \
    --mapfile ${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_FILENAME}