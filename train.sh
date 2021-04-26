python3.7 train_supervised_text_embedding.py \
    --train-file $GEMINI_DATA_IN/sbert.small/sbert.train.txt \
    --eval-file $GEMINI_DATA_IN/sbert.small/sbert.eval.txt \
    --epoch 5 \
    --batch-size 32 |&tee train.log 