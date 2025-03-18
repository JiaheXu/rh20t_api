SCENE_FOLDER=/media/jiahe/data/RH20T_rgb_resized/RH20T_cfg3/task_0015_user_0010_scene_0010_cfg_0003
CACHE_FOLDER=cache/
#python3 visualize.py --scene_folder $SCENE_FOLDER --cache_folder $CACHE_FOLDER --preprocess
python3 extract_rgbd.py --scene_folder $SCENE_FOLDER --cache_folder $CACHE_FOLDER #--preprocess
