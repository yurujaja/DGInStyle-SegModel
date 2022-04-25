# Obtained from: https://github.com/lhoyer/HRDA
# Modification:
# - Test on all five real-world datasets

#!/bin/bash

TEST_ROOT=$1
CONFIG_FILE="${TEST_ROOT}/*${TEST_ROOT: -1}.json"
CHECKPOINT_FILE="${TEST_ROOT}/latest.pth"
SHOW_DIR="${TEST_ROOT}/preds"
echo 'Config File:' $CONFIG_FILE
echo 'Checkpoint File:' $CHECKPOINT_FILE
echo 'Predictions Output Directory:' $SHOW_DIR

python -m tools.test ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mIoU --dataset Cityscapes
python -m tools.test ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mIoU --dataset BDD100K
python -m tools.test ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mIoU --dataset Mapillary --eval-option efficient_test=True
python -m tools.test ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mIoU --dataset ACDC
python -m tools.test ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mIoU --dataset DarkZurich