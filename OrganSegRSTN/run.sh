####################################################################################################
# RSTN: Recurrent Saliency Transformation Network for organ segmentation framework                 #
#                                                                                                  #
# If you use our codes, please cite our paper accordingly:                                         #
#     Qihang Yu, Lingxi Xie, Yan Wang, Yuyin Zhou, Elliot K. Fishman, Alan L. Yuille,              #
#         "Recurrent Saliency Transformation Network:                                              #
#             Incorporating Multi-Stage Visual Cues for Small Organ Segmentation",                 #
#         in IEEE Conference on Computer Vision and Pattern Recognition, 2018.                     #
#                                                                                                  #
# NOTE: this program can be used for multi-organ segmentation.                                     #
#     Please also refer to its previous version, OrganSegC2F.                                      #
####################################################################################################

####################################################################################################
# variables for convenience
CURRENT_ORGAN_ID=1
CURRENT_PLANE=A
CURRENT_FOLD=0
CURRENT_GPU=$CURRENT_FOLD

####################################################################################################
# turn on these swithes to execute each module
ENABLE_INITIALIZATION=0
ENABLE_INDIV_TRAINING=0
ENABLE_JOINT_TRAINING=0
ENABLE_COARSE_TESTING=0
ENABLE_COARSE_FUSION=0
ENABLE_ORACLE_TESTING=0
ENABLE_ORACLE_FUSION=0
ENABLE_COARSE2FINE_TESTING=0
# indiv_training settings: X|Y|Z
INDIV_TRAINING_ORGAN_ID=$CURRENT_ORGAN_ID
INDIV_TRAINING_PLANE=$CURRENT_PLANE
INDIV_TRAINING_GPU=$CURRENT_GPU
# joint_training settings: X|Y|Z
JOINT_TRAINING_ORGAN_ID=$CURRENT_ORGAN_ID
JOINT_TRAINING_PLANE=$CURRENT_PLANE
JOINT_TRAINING_GPU=$CURRENT_GPU
# coarse_testing settings: X|Y|Z, before this, coarse-scaled models shall be ready
COARSE_TESTING_ORGAN_ID=$CURRENT_ORGAN_ID
COARSE_TESTING_PLANE=$CURRENT_PLANE
COARSE_TESTING_GPU=$CURRENT_GPU
# coarse_fusion settings: before this, coarse-scaled results on 3 views shall be ready
COARSE_FUSION_ORGAN_ID=$CURRENT_ORGAN_ID
# oracle_testing settings: X|Y|Z, before this, fine-scaled models shall be ready
ORACLE_TESTING_ORGAN_ID=$CURRENT_ORGAN_ID
ORACLE_TESTING_PLANE=$CURRENT_PLANE
ORACLE_TESTING_GPU=$CURRENT_GPU
# oracle_fusion settings: before this, fine-scaled results on 3 views shall be ready
ORACLE_FUSION_ORGAN_ID=$CURRENT_ORGAN_ID
# fine_testing settings: before this, both coarse-scaled and fine-scaled models shall be ready
COARSE2FINE_TESTING_ORGAN_ID=$CURRENT_ORGAN_ID
COARSE2FINE_TESTING_GPU=$CURRENT_GPU


####################################################################################################
# defining the root path which stores image and label data
DATA_PATH='/media/Med_2T2/data2/'
LIB_PATH='/media/Med_2T2/data2/'
mkdir ${DATA_PATH}logs/

####################################################################################################
# export PYTHONPATH (related to your path to CAFFE)
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=${LIB_PATH}libs/caffe-master/python:$PYTHONPATH

####################################################################################################
# data initialization: only needs to be run once
# variables
ORGAN_NUMBER=1
FOLDS=4
LOW_RANGE=-100
HIGH_RANGE=240
# init.py : data_path, organ_number, folds, low_range, high_range
if [ "$ENABLE_INITIALIZATION" = "1" ]
then
    python init.py \
        $DATA_PATH $ORGAN_NUMBER $FOLDS $LOW_RANGE $HIGH_RANGE
fi

####################################################################################################
# the individual and joint training processes
# variables
SLICE_THRESHOLD=0.98
SLICE_THICKNESS=3
LEARNING_RATE1=1e-5
LEARNING_RATE2=1e-5
LEARNING_RATE_M1=10
LEARNING_RATE_M2=10
TRAINING_MARGIN=20
TRAINING_PROB=0.5
TRAINING_SAMPLE_BATCH=1
TRAINING_STEP=10000
TRAINING_MAX_ITERATIONS1=40000
TRAINING_MAX_ITERATIONS2=20000
TRAINING_FRACTION=0.25
TRAINING_TOTAL_ITERATIONS=$(($TRAINING_MAX_ITERATIONS1+$TRAINING_MAX_ITERATIONS2))
# individual training
if [ "$ENABLE_INDIV_TRAINING" = "1" ]
then
    INDIV_TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
else
    INDIV_TIMESTAMP=_
fi
# indiv_training.py : data_path, current_fold, organ_number, low_range, high_range,
#     slice_threshold, slice_thickness, organ_ID, plane, GPU_ID,
#     learning_rate1, learning_rate2 (not used), margin, prob, sample_batch,
#     step, max_iterations1, max_iterations2 (not used), fraction, timestamp
if [ "$ENABLE_INDIV_TRAINING" = "1" ]
then
    if [ "$INDIV_TRAINING_PLANE" = "X" ] || [ "$INDIV_TRAINING_PLANE" = "A" ]
    then
        INDIV_MODELNAME=XI${SLICE_THICKNESS}_${INDIV_TRAINING_ORGAN_ID}
        INDIV_LOG=${DATA_PATH}logs/FD${CURRENT_FOLD}:${INDIV_MODELNAME}_${INDIV_TIMESTAMP}.txt
        python indiv_training.py \
            $DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE \
            $SLICE_THRESHOLD $SLICE_THICKNESS \
            $INDIV_TRAINING_ORGAN_ID X $INDIV_TRAINING_GPU \
            $LEARNING_RATE1 $LEARNING_RATE_M1 $LEARNING_RATE2 $LEARNING_RATE_M2 \
            $TRAINING_MARGIN $TRAINING_PROB $TRAINING_SAMPLE_BATCH \
            $TRAINING_STEP $TRAINING_MAX_ITERATIONS1 $TRAINING_MAX_ITERATIONS2 \
            $TRAINING_FRACTION $INDIV_TIMESTAMP 1 2>&1 | tee $INDIV_LOG
    fi
    if [ "$INDIV_TRAINING_PLANE" = "Y" ] || [ "$INDIV_TRAINING_PLANE" = "A" ]
    then
        INDIV_MODELNAME=YI${SLICE_THICKNESS}_${INDIV_TRAINING_ORGAN_ID}
        INDIV_LOG=${DATA_PATH}logs/FD${CURRENT_FOLD}:${INDIV_MODELNAME}_${INDIV_TIMESTAMP}.txt
        python indiv_training.py \
            $DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE \
            $SLICE_THRESHOLD $SLICE_THICKNESS \
            $INDIV_TRAINING_ORGAN_ID Y $INDIV_TRAINING_GPU \
            $LEARNING_RATE1 $LEARNING_RATE_M1 $LEARNING_RATE2 $LEARNING_RATE_M2 \
            $TRAINING_MARGIN $TRAINING_PROB $TRAINING_SAMPLE_BATCH \
            $TRAINING_STEP $TRAINING_MAX_ITERATIONS1 $TRAINING_MAX_ITERATIONS2 \
            $TRAINING_FRACTION $INDIV_TIMESTAMP 1 2>&1 | tee $INDIV_LOG
    fi
    if [ "$INDIV_TRAINING_PLANE" = "Z" ] || [ "$INDIV_TRAINING_PLANE" = "A" ]
    then
        INDIV_MODELNAME=ZI${SLICE_THICKNESS}_${INDIV_TRAINING_ORGAN_ID}
        INDIV_LOG=${DATA_PATH}logs/FD${CURRENT_FOLD}:${INDIV_MODELNAME}_${INDIV_TIMESTAMP}.txt
        python indiv_training.py \
            $DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE \
            $SLICE_THRESHOLD $SLICE_THICKNESS \
            $INDIV_TRAINING_ORGAN_ID Z $INDIV_TRAINING_GPU \
            $LEARNING_RATE1 $LEARNING_RATE_M1 $LEARNING_RATE2 $LEARNING_RATE_M2 \
            $TRAINING_MARGIN $TRAINING_PROB $TRAINING_SAMPLE_BATCH \
            $TRAINING_STEP $TRAINING_MAX_ITERATIONS1 $TRAINING_MAX_ITERATIONS2 \
            $TRAINING_FRACTION $INDIV_TIMESTAMP 1 2>&1 | tee $INDIV_LOG
    fi
fi
# joint training
JOINT_TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
# joint_training.py : data_path, current_fold, organ_number, low_range, high_range,
#     slice_threshold, slice_thickness, organ_ID, plane, GPU_ID,
#     learning_rate1, learning_rate2, margin, prob, sample_batch,
#     step, max_iterations1, max_iterations2, timestamp
if [ "$ENABLE_JOINT_TRAINING" = "1" ]
then
    if [ "$JOINT_TRAINING_PLANE" = "X" ] || [ "$JOINT_TRAINING_PLANE" = "A" ]
    then
        JOINT_MODELNAME=XJ${SLICE_THICKNESS}_${JOINT_TRAINING_ORGAN_ID}
        JOINT_LOG=${DATA_PATH}logs/FD${CURRENT_FOLD}:${JOINT_MODELNAME}_${JOINT_TIMESTAMP}.txt
        python joint_training.py \
            $DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE \
            $SLICE_THRESHOLD $SLICE_THICKNESS \
            $JOINT_TRAINING_ORGAN_ID X $JOINT_TRAINING_GPU \
            $LEARNING_RATE1 $LEARNING_RATE_M1 $LEARNING_RATE2 $LEARNING_RATE_M2 \
            $TRAINING_MARGIN $TRAINING_PROB $TRAINING_SAMPLE_BATCH \
            $TRAINING_STEP $TRAINING_MAX_ITERATIONS1 $TRAINING_MAX_ITERATIONS2 \
            $INDIV_TIMESTAMP $JOINT_TIMESTAMP 2>&1 | tee $JOINT_LOG
    fi
    if [ "$JOINT_TRAINING_PLANE" = "Y" ] || [ "$JOINT_TRAINING_PLANE" = "A" ]
    then
        JOINT_MODELNAME=YJ${SLICE_THICKNESS}_${JOINT_TRAINING_ORGAN_ID}
        JOINT_LOG=${DATA_PATH}logs/FD${CURRENT_FOLD}:${JOINT_MODELNAME}_${JOINT_TIMESTAMP}.txt
        python joint_training.py \
            $DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE \
            $SLICE_THRESHOLD $SLICE_THICKNESS \
            $JOINT_TRAINING_ORGAN_ID Y $JOINT_TRAINING_GPU \
            $LEARNING_RATE1 $LEARNING_RATE_M1 $LEARNING_RATE2 $LEARNING_RATE_M2 \
            $TRAINING_MARGIN $TRAINING_PROB $TRAINING_SAMPLE_BATCH \
            $TRAINING_STEP $TRAINING_MAX_ITERATIONS1 $TRAINING_MAX_ITERATIONS2 \
            $INDIV_TIMESTAMP $JOINT_TIMESTAMP 2>&1 | tee $JOINT_LOG
    fi
    if [ "$JOINT_TRAINING_PLANE" = "Z" ] || [ "$JOINT_TRAINING_PLANE" = "A" ]
    then
        JOINT_MODELNAME=ZJ${SLICE_THICKNESS}_${JOINT_TRAINING_ORGAN_ID}
        JOINT_LOG=${DATA_PATH}logs/FD${CURRENT_FOLD}:${JOINT_MODELNAME}_${JOINT_TIMESTAMP}.txt
        python joint_training.py \
            $DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE \
            $SLICE_THRESHOLD $SLICE_THICKNESS \
            $JOINT_TRAINING_ORGAN_ID Z $JOINT_TRAINING_GPU \
            $LEARNING_RATE1 $LEARNING_RATE_M1 $LEARNING_RATE2 $LEARNING_RATE_M2 \
            $TRAINING_MARGIN $TRAINING_PROB $TRAINING_SAMPLE_BATCH \
            $TRAINING_STEP $TRAINING_MAX_ITERATIONS1 $TRAINING_MAX_ITERATIONS2 \
            $INDIV_TIMESTAMP $JOINT_TIMESTAMP 2>&1 | tee $JOINT_LOG
    fi
fi

####################################################################################################
# the coarse-scaled testing processes
# variables
COARSE_TESTING_STARTING_ITERATIONS=$TRAINING_TOTAL_ITERATIONS
COARSE_TESTING_STEP=$TRAINING_STEP
COARSE_TESTING_MAX_ITERATIONS=$TRAINING_TOTAL_ITERATIONS
COARSE_TIMESTAMP1=_
COARSE_TIMESTAMP2=_
# coarse_testing.py : data_path, current_fold, organ_number, low_range, high_range,
#     slice_threshold, slice_thickness, organ_ID, plane, GPU_ID,
#     learning_rate1, learning_rate2, margin, prob, sample_batch,
#     step, max_iterations1, max_iterations2,
#     starting_iterations, step, max_iterations,
#     timestamp1 (optional), timestamp2 (optional)
if [ "$ENABLE_COARSE_TESTING" = "1" ]
then
    if [ "$COARSE_TESTING_PLANE" = "X" ] || [ "$COARSE_TESTING_PLANE" = "A" ]
    then
        python coarse_testing.py \
            $DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE \
            $SLICE_THRESHOLD $SLICE_THICKNESS \
            $COARSE_TESTING_ORGAN_ID X $COARSE_TESTING_GPU \
            $LEARNING_RATE1 $LEARNING_RATE_M1 $LEARNING_RATE2 $LEARNING_RATE_M2 \
            $TRAINING_MARGIN $TRAINING_PROB $TRAINING_SAMPLE_BATCH \
            $TRAINING_MAX_ITERATIONS1 $TRAINING_MAX_ITERATIONS2 \
            $COARSE_TESTING_STARTING_ITERATIONS $COARSE_TESTING_STEP \
            $COARSE_TESTING_MAX_ITERATIONS \
            $COARSE_TIMESTAMP1 $COARSE_TIMESTAMP2
    fi
    if [ "$COARSE_TESTING_PLANE" = "Y" ] || [ "$COARSE_TESTING_PLANE" = "A" ]
    then
        python coarse_testing.py \
            $DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE \
            $SLICE_THRESHOLD $SLICE_THICKNESS \
            $COARSE_TESTING_ORGAN_ID Y $COARSE_TESTING_GPU \
            $LEARNING_RATE1 $LEARNING_RATE_M1 $LEARNING_RATE2 $LEARNING_RATE_M2 \
            $TRAINING_MARGIN $TRAINING_PROB $TRAINING_SAMPLE_BATCH \
            $TRAINING_MAX_ITERATIONS1 $TRAINING_MAX_ITERATIONS2 \
            $COARSE_TESTING_STARTING_ITERATIONS $COARSE_TESTING_STEP \
            $COARSE_TESTING_MAX_ITERATIONS \
            $COARSE_TIMESTAMP1 $COARSE_TIMESTAMP2
    fi
    if [ "$COARSE_TESTING_PLANE" = "Z" ] || [ "$COARSE_TESTING_PLANE" = "A" ]
    then
        python coarse_testing.py \
            $DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE \
            $SLICE_THRESHOLD $SLICE_THICKNESS \
            $COARSE_TESTING_ORGAN_ID Z $COARSE_TESTING_GPU \
            $LEARNING_RATE1 $LEARNING_RATE_M1 $LEARNING_RATE2 $LEARNING_RATE_M2 \
            $TRAINING_MARGIN $TRAINING_PROB $TRAINING_SAMPLE_BATCH \
            $TRAINING_MAX_ITERATIONS1 $TRAINING_MAX_ITERATIONS2 \
            $COARSE_TESTING_STARTING_ITERATIONS $COARSE_TESTING_STEP \
            $COARSE_TESTING_MAX_ITERATIONS \
            $COARSE_TIMESTAMP1 $COARSE_TIMESTAMP2
    fi
fi

####################################################################################################
# the coarse-scaled fusion process
# variables
COARSE_FUSION_STARTING_ITERATIONS=$TRAINING_TOTAL_ITERATIONS
COARSE_FUSION_STEP=$TRAINING_STEP
COARSE_FUSION_MAX_ITERATIONS=$TRAINING_TOTAL_ITERATIONS
COARSE_FUSION_THRESHOLD=0.5
COARSE_TIMESTAMP1_X=_
COARSE_TIMESTAMP1_Y=_
COARSE_TIMESTAMP1_Z=_
COARSE_TIMESTAMP2_X=_
COARSE_TIMESTAMP2_Y=_
COARSE_TIMESTAMP2_Z=_
# coarse_fusion.py : data_path, current_fold, organ_number, low_range, high_range,
#     slice_threshold, slice_thickness, organ_ID, plane, GPU_ID,
#     learning_rate1, learning_rate_m1, learning_rate2, learning_rate_m2, margin,
#     starting_iterations, step, max_iterations, threshold,
#     timestamp1_X (optional), timestamp1_Y (optional), timestamp1_Z (optional),
#     timestamp2_X (optional), timestamp2_Y (optional), timestamp2_Z (optional)
if [ "$ENABLE_COARSE_FUSION" = "1" ]
then
    python coarse_fusion.py \
        $DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE \
        $SLICE_THRESHOLD $SLICE_THICKNESS $COARSE_TESTING_ORGAN_ID $COARSE_TESTING_GPU \
        $LEARNING_RATE1 $LEARNING_RATE_M1 $LEARNING_RATE2 $LEARNING_RATE_M2 $TRAINING_MARGIN \
        $COARSE_FUSION_STARTING_ITERATIONS $COARSE_FUSION_STEP \
        $COARSE_FUSION_MAX_ITERATIONS $COARSE_FUSION_THRESHOLD \
        $COARSE_TIMESTAMP1_X $COARSE_TIMESTAMP1_Y $COARSE_TIMESTAMP1_Z \
        $COARSE_TIMESTAMP2_X $COARSE_TIMESTAMP2_Y $COARSE_TIMESTAMP2_Z
fi

####################################################################################################
# the oracle testing processes
# variables
ORACLE_TESTING_STARTING_ITERATIONS=$TRAINING_TOTAL_ITERATIONS
ORACLE_TESTING_STEP=$TRAINING_STEP
ORACLE_TESTING_MAX_ITERATIONS=$TRAINING_TOTAL_ITERATIONS
ORACLE_TIMESTAMP1=_
ORACLE_TIMESTAMP2=_
# oracle_testing.py : data_path, current_fold, organ_number, low_range, high_range,
#     slice_threshold, slice_thickness, organ_ID, plane, GPU_ID,
#     learning_rate1, learning_rate_m1, learning_rate2, learning_rate_m2,
#     margin, prob, sample_batch,
#     step, max_iterations1, max_iterations2,
#     starting_iterations, step, max_iterations,
#     timestamp1 (optional), timestamp2 (optional)
if [ "$ENABLE_ORACLE_TESTING" = "1" ]
then
    if [ "$ORACLE_TESTING_PLANE" = "X" ] || [ "$ORACLE_TESTING_PLANE" = "A" ]
    then
        python oracle_testing.py \
            $DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE \
            $SLICE_THRESHOLD $SLICE_THICKNESS \
            $ORACLE_TESTING_ORGAN_ID X $ORACLE_TESTING_GPU \
            $LEARNING_RATE1 $LEARNING_RATE_M1 $LEARNING_RATE2 $LEARNING_RATE_M2 \
            $TRAINING_MARGIN $TRAINING_PROB $TRAINING_SAMPLE_BATCH \
            $TRAINING_MAX_ITERATIONS1 $TRAINING_MAX_ITERATIONS2 \
            $ORACLE_TESTING_STARTING_ITERATIONS $ORACLE_TESTING_STEP \
            $ORACLE_TESTING_MAX_ITERATIONS \
            $ORACLE_TIMESTAMP1 $ORACLE_TIMESTAMP2
    fi
    if [ "$ORACLE_TESTING_PLANE" = "Y" ] || [ "$ORACLE_TESTING_PLANE" = "A" ]
    then
        python oracle_testing.py \
            $DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE \
            $SLICE_THRESHOLD $SLICE_THICKNESS \
            $ORACLE_TESTING_ORGAN_ID Y $ORACLE_TESTING_GPU \
            $LEARNING_RATE1 $LEARNING_RATE_M1 $LEARNING_RATE2 $LEARNING_RATE_M2 \
            $TRAINING_MARGIN $TRAINING_PROB $TRAINING_SAMPLE_BATCH \
            $TRAINING_MAX_ITERATIONS1 $TRAINING_MAX_ITERATIONS2 \
            $ORACLE_TESTING_STARTING_ITERATIONS $ORACLE_TESTING_STEP \
            $ORACLE_TESTING_MAX_ITERATIONS \
            $ORACLE_TIMESTAMP1 $ORACLE_TIMESTAMP2
    fi
    if [ "$ORACLE_TESTING_PLANE" = "Z" ] || [ "$ORACLE_TESTING_PLANE" = "A" ]
    then
        python oracle_testing.py \
            $DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE \
            $SLICE_THRESHOLD $SLICE_THICKNESS \
            $ORACLE_TESTING_ORGAN_ID Z $ORACLE_TESTING_GPU \
            $LEARNING_RATE1 $LEARNING_RATE_M1 $LEARNING_RATE2 $LEARNING_RATE_M2 \
            $TRAINING_MARGIN $TRAINING_PROB $TRAINING_SAMPLE_BATCH \
            $TRAINING_MAX_ITERATIONS1 $TRAINING_MAX_ITERATIONS2 \
            $ORACLE_TESTING_STARTING_ITERATIONS $ORACLE_TESTING_STEP \
            $ORACLE_TESTING_MAX_ITERATIONS \
            $ORACLE_TIMESTAMP1 $ORACLE_TIMESTAMP2
    fi
fi

####################################################################################################
# the oracle-scaled fusion process
# variables
ORACLE_FUSION_STARTING_ITERATIONS=$TRAINING_TOTAL_ITERATIONS
ORACLE_FUSION_STEP=$TRAINING_STEP
ORACLE_FUSION_MAX_ITERATIONS=$TRAINING_TOTAL_ITERATIONS
ORACLE_FUSION_THRESHOLD=0.5
ORACLE_TIMESTAMP1_X=_
ORACLE_TIMESTAMP1_Y=_
ORACLE_TIMESTAMP1_Z=_
ORACLE_TIMESTAMP2_X=_
ORACLE_TIMESTAMP2_Y=_
ORACLE_TIMESTAMP2_Z=_
# oracle_fusion.py : data_path, current_fold, organ_number, low_range, high_range,
#     slice_threshold, slice_thickness, organ_ID, plane, GPU_ID,
#     learning_rate1, learning_rate_m1, learning_rate2, learning_rate_m2, margin,
#     starting_iterations, step, max_iterations, threshold,
#     timestamp1_X (optional), timestamp1_Y (optional), timestamp1_Z (optional),
#     timestamp2_X (optional), timestamp2_Y (optional), timestamp2_Z (optional)
if [ "$ENABLE_ORACLE_FUSION" = "1" ]
then
    python oracle_fusion.py \
        $DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE \
        $SLICE_THRESHOLD $SLICE_THICKNESS $ORACLE_TESTING_ORGAN_ID $ORACLE_TESTING_GPU \
        $LEARNING_RATE1 $LEARNING_RATE_M1 $LEARNING_RATE2 $LEARNING_RATE_M2 $TRAINING_MARGIN \
        $ORACLE_FUSION_STARTING_ITERATIONS $ORACLE_FUSION_STEP \
        $ORACLE_FUSION_MAX_ITERATIONS $ORACLE_FUSION_THRESHOLD \
        $ORACLE_TIMESTAMP1_X $ORACLE_TIMESTAMP1_Y $ORACLE_TIMESTAMP1_Z \
        $ORACLE_TIMESTAMP2_X $ORACLE_TIMESTAMP2_Y $ORACLE_TIMESTAMP2_Z
fi

####################################################################################################
# the coarse-to-fine testing process
# variables
COARSE_FUSION_STARTING_ITERATIONS=$TRAINING_TOTAL_ITERATIONS
COARSE_FUSION_STEP=$TRAINING_STEP
COARSE_FUSION_MAX_ITERATIONS=$TRAINING_TOTAL_ITERATIONS
COARSE_FUSION_THRESHOLD=0.5
FINE_TESTING_STARTING_ITERATIONS=$TRAINING_TOTAL_ITERATIONS
FINE_TESTING_STEP=$TRAINING_STEP
FINE_TESTING_MAX_ITERATIONS=$TRAINING_TOTAL_ITERATIONS
FINE_FUSION_THRESHOLD=0.5
COARSE2FINE_TIMESTAMP1_X=_
COARSE2FINE_TIMESTAMP1_Y=_
COARSE2FINE_TIMESTAMP1_Z=_
COARSE2FINE_TIMESTAMP2_X=_
COARSE2FINE_TIMESTAMP2_Y=_
COARSE2FINE_TIMESTAMP2_Z=_
MAX_ROUNDS=10
# coarse2fine_testing.py : data_path, current_fold, organ_number, low_range, high_range,
#     slice_threshold, slice_thickness, organ_ID, GPU_ID,
#     learning_rate1, learning_rate_m1, learning_rate2, learning_rate_m2, margin,
#     coarse_fusion_starting_iterations, coarse_fusion_step, coarse_fusion_max_iterations,
#     coarse_fusion_threshold, coarse_fusion_code,
#     fine_starting_iterations, fine_step, fine_max_iterations,
#     fine_fusion_threshold, max_rounds
#     timestamp1_X (optional), timestamp1_Y (optional), timestamp1_Z (optional),
#     timestamp2_X (optional), timestamp2_Y (optional), timestamp2_Z (optional)
if [ "$ENABLE_COARSE2FINE_TESTING" = "1" ]
then
    python coarse2fine_testing.py \
        $DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE \
        $SLICE_THRESHOLD $SLICE_THICKNESS $COARSE2FINE_TESTING_ORGAN_ID $COARSE2FINE_TESTING_GPU \
        $LEARNING_RATE1 $LEARNING_RATE_M1 $LEARNING_RATE2 $LEARNING_RATE_M2 $TRAINING_MARGIN \
        $COARSE_FUSION_STARTING_ITERATIONS $COARSE_FUSION_STEP $COARSE_FUSION_MAX_ITERATIONS \
        $COARSE_FUSION_THRESHOLD \
        $FINE_TESTING_STARTING_ITERATIONS $FINE_TESTING_STEP $FINE_TESTING_MAX_ITERATIONS \
        $FINE_FUSION_THRESHOLD $MAX_ROUNDS \
        $COARSE2FINE_TIMESTAMP1_X $COARSE2FINE_TIMESTAMP1_Y $COARSE2FINE_TIMESTAMP1_Z \
        $COARSE2FINE_TIMESTAMP2_X $COARSE2FINE_TIMESTAMP2_Y $COARSE2FINE_TIMESTAMP2_Z
fi

####################################################################################################
