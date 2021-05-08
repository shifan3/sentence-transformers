python3.7 train.py \
    --train-file $GEMINI_DATA_IN/sbert.2/train.txt \
    --eval-file $GEMINI_DATA_IN/sbert.2/eval.txt \
    --output-dir $GEMINI_DATA_OUT/ \
    --epoch 10 \
    --eval-log-file eval.log \
    --batch-size 32 > train.log 2>&1