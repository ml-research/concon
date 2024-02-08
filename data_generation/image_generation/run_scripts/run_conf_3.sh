#!/bin/bash

# path to your blender executable
blender=/usr/bin/blender-2.78c-linux-glibc219-x86_64/blender

#----------------------------------------------------------#
NUM_TRAIN_SAMPLES=3000
NUM_VAL_SAMPLES=750
NUM_TEST_SAMPLES=750

FLAG_VAL="False"
FLAG_TEST="False"

NUM_PARALLEL_THREADS=40
NUM_THREADS=4
MIN_OBJECTS=4
MAX_OBJECTS=4
MAX_RETRIES=30

IMG_NAME_PREFIX='img_conf'
IMG_TYPES=(0 1)

MIN_PIXELS_PER_OBJECT=200
DATASET_TYPE='strict_'$MIN_PIXELS_PER_OBJECT
CONF_FILENAME='IF-conf.json'

TASK_IDS=(0 1 2)

STRICT=1


#----------------------------------------------------------#
# # generate training images
for IMG_TYPE in ${IMG_TYPES[@]}; do

    if [ "$IMG_TYPE" = 0 ]; then
        IF_type="False"
    else 
        IF_type="True"
    fi

    for TASK_ID in ${TASK_IDS[@]}; do
        time $blender \
                --threads $NUM_THREADS \
                --background -noaudio \
                --python render_images_IF.py \
                -- --output_image_dir ../output/$DATASET_TYPE/train/images/t$TASK_ID/$IMG_TYPE \
                    --output_scene_dir ../output/$DATASET_TYPE/train/scenes/t$TASK_ID/$IMG_TYPE \
                    --output_scene_file ../output/$DATASET_TYPE/train/thesis/thesis_scenes_train.json \
                    --strict $STRICT \
                    --filename_prefix $IMG_NAME_PREFIX \
                    --max_retries $MAX_RETRIES \
                    --num_images $NUM_TRAIN_SAMPLES \
                    --min_objects $MIN_OBJECTS \
                    --max_objects $MAX_OBJECTS \
                    --num_parallel_threads $NUM_PARALLEL_THREADS \
                    --width 224 --height 224 \
                    --properties_json data/properties.json \
                    --min_pixels_per_object $MIN_PIXELS_PER_OBJECT \
                    --conf_class_combos_json data/$CONF_FILENAME \
                    --gt_class_combos_json data/IF-gt.json \
                    --img_class_id $TASK_ID \
                    --IF_type $IF_type \
                    --validation 'False' \

    done
done

# # merge all classes join files to one json file
# python merge_json_files.py --json_dir ./output/train/confounder4_F/

# #----------------------------------------------------------#

# generate test images
for IMG_TYPE in ${IMG_TYPES[@]}; do

    if [ "$IMG_TYPE" = 0 ]; then
        IF_type="False"
    else 
        IF_type="True"
    fi
    for TASK_ID in ${TASK_IDS[@]}; do
        time $blender \
                --threads $NUM_THREADS \
                --background -noaudio \
                --python render_images_IF.py \
                -- --output_image_dir ../output/$DATASET_TYPE/test/images/t$TASK_ID/$IMG_TYPE  \
                    --output_scene_dir ../output/$DATASET_TYPE/test/scenes/t$TASK_ID/$IMG_TYPE  \
                    --output_scene_file ../output/$DATASET_TYPE/test/thesis/thesis_scenes_test.json \
                    --filename_prefix $IMG_NAME_PREFIX \
                    --strict $STRICT \
                    --max_retries $MAX_RETRIES \
                    --num_images $NUM_TEST_SAMPLES \
                    --min_objects $MIN_OBJECTS \
                    --max_objects $MAX_OBJECTS \
                    --num_parallel_threads $NUM_PARALLEL_THREADS \
                    --width 224 --height 224 \
                    --properties_json data/properties.json \
                    --min_pixels_per_object $MIN_PIXELS_PER_OBJECT \
                    --conf_class_combos_json data/$CONF_FILENAME \
                    --gt_class_combos_json data/IF-gt.json \
                    --img_class_id $TASK_ID \
                    --IF_type $IF_type \
                    --validation 'False' \

    done
done

# # # # # merge all classes join files to one json file
# # # # python merge_json_files.py --json_dir ./output/test/confounder4_F/

# # # # #----------------------------------------------------------#

# generate confounded val images
for IMG_TYPE in ${IMG_TYPES[@]}; do

    if [ "$IMG_TYPE" = 0 ]; then
        IF_type="False"
    else 
        IF_type="True"
    fi

    for TASK_ID in ${TASK_IDS[@]}; do
        time $blender \
                --threads $NUM_THREADS \
                --background -noaudio \
                --python render_images_IF.py \
                -- --output_image_dir ../output/$DATASET_TYPE/val/images/t$TASK_ID/$IMG_TYPE  \
                    --output_scene_dir ../output/$DATASET_TYPE/val/scenes/t$TASK_ID/$IMG_TYPE  \
                    --output_scene_file ../output/$DATASET_TYPE/val/thesis/thesis_scenes_val.json \
                    --filename_prefix $IMG_NAME_PREFIX \
                    --strict $STRICT \
                    --max_retries $MAX_RETRIES \
                    --num_images $NUM_VAL_SAMPLES \
                    --min_objects $MIN_OBJECTS \
                    --max_objects $MAX_OBJECTS \
                    --num_parallel_threads $NUM_PARALLEL_THREADS \
                    --width 224 --height 224 \
                    --properties_json data/properties.json \
                    --min_pixels_per_object $MIN_PIXELS_PER_OBJECT \
                    --conf_class_combos_json data/$CONF_FILENAME \
                    --gt_class_combos_json data/IF-gt.json \
                    --img_class_id $TASK_ID \
                    --IF_type $IF_type \
                    --validation 'False' \

    done
done

# # # merge all classes join files to one json file
# # # python merge_json_files.py --json_dir ./output/val/confounder4_F/
