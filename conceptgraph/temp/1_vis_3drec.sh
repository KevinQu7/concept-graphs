# office 0-4
# room 0-2

SCENE_NAME=office0
python scripts/run_slam_rgb.py \
    --dataset_root $REPLICA_ROOT \
    --dataset_config $REPLICA_CONFIG_PATH \
    --scene_id $SCENE_NAME \
    --image_height 480 \
    --image_width 640 \
    --stride 10 \
    --visualize