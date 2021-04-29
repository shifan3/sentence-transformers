python3.7 train.py \
    --train-file sbert.train.txt \
    --eval-file sbert.eval.txt \
    --output-dir $GEMINI_DATA_OUT/ \
    --epoch 5 \
    --eval-log-file eval.log \
    --batch-size 32 > train.log 2>&1