#!/bin/bash
EXEC=project1_executable.py
SET1=data_sets1/data_sets1
SET2=data_sets2/data_sets2
TRAIN=training_set.csv
VALIDATE=validation_set.csv
TEST=test_set.csv
PRINT=no
RES_DIR=results
mkdir -p $RES_DIR
TIMESTAMP=$(date +%s)
RES_LOG1=set1_results$TIMESTAMP.txt
RES_LOG2=set2_results$TIMESTAMP.txt

echo 'Results for Dataset 1' > $RES_DIR/$RES_LOG1
echo 'Params        Pre-Prune Accuracy          Post-Prune Accuracy' >> $RES_DIR/$RES_LOG1
echo 'L     K       VarImp      InfoGain        VarImp      InfoGain' >> $RES_DIR/$RES_LOG1
echo 'Results for Dataset 2' > $RES_DIR/$RES_LOG2
echo 'Params        Pre-Prune Accuracy          Post-Prune Accuracy' >> $RES_DIR/$RES_LOG2
echo 'L     K       VarImp      InfoGain        VarImp      InfoGain' >> $RES_DIR/$RES_LOG2

for L in {1,2,4,8,16}
do
  for K in {1,2,4,8,16}
  do
    python3 $EXEC $L $K $SET1/$TRAIN $SET1/$VALIDATE $SET1/$TEST $PRINT >> $RES_DIR/$RES_LOG1
    python3 $EXEC $L $K $SET2/$TRAIN $SET2/$VALIDATE $SET2/$TEST $PRINT >> $RES_DIR/$RES_LOG2
  done
done
