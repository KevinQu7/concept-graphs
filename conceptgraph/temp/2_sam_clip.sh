SCENE_NAME=office3

# The CoceptGraphs (without open-vocab detector)
python scripts/generate_gsa_results.py \
    --dataset_root $REPLICA_ROOT \
    --dataset_config $REPLICA_CONFIG_PATH \
    --scene_id $SCENE_NAME \
    --class_set none \
    --stride 5