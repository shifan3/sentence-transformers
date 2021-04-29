python3.7 train.py \
    --train-file $GEMINI_DATA_IN/sbert.small/sbert.train.txt \
    --eval-file $GEMINI_DATA_IN/sbert.small/sbert.eval.txt \
    --output-dir $GEMINI_DATA_OUT/
    --epoch 5 \
    --eval-log-file $GEMINI_DATA_OUT/eval.log \
    --batch-size 32 > train.log 2>&1