import codecs
import random
import re
import tqdm
import transformers
import sys
from sentence_transformers.readers import InputExample
from sentence_transformers import SentenceTransformer, models, losses
from sentence_transformers.evaluation.SimilarityFunction import SimilarityFunction
from torch import nn
import torch
import os
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
import torch
from sentence_transformers import evaluation
from optparse import OptionParser

parser = OptionParser()
parser.add_option("", "--train-file", type=str, default='sbert.train.txt', dest = 'train_file')
parser.add_option("", "--eval-file", type=str, default='sbert.eval.txt', dest = 'eval_file')
parser.add_option("", "--output-dir", type=str, default=None, dest = 'output_dir')
parser.add_option("", "--log-file", type=str, default=None, dest = 'log_file')
parser.add_option("", "--batch-size", type=int, default=8, dest = 'batch_size')
parser.add_option("", "--epoch", type=int, default=5, dest = 'epoch')

#parser.add_option("", "--save-dir", type=str, default='/mnt/data5/feat_test', dest = 'save_dir')

opts, args = parser.parse_args()


f_log = sys.stderr
if opts.log_file:
    f_log = codecs.open(opts.log_file, 'w', 'utf-8')

word_embedding_model = models.Transformer('hfl/chinese-bert-wwm-ext', max_seq_length=256)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())


dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), 
    out_features=1024, activation_function=nn.Tanh())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
model.cuda()

train_examples = []
eval_s1 = []
eval_s2 = []
eval_score = []
with codecs.open(opts.train_file, 'r', 'utf-8') as f:
    t1 = None
    t2 = None
    simi = None
    for i, line in enumerate(tqdm.tqdm(f, file=f_log)):
        if i % 3 == 0:
            t1 = line.strip()
        elif i % 3 == 1:
            t2 = line.strip()
        elif i % 3 == 2:
            simi = float(line.strip())
            t1 = ' '.join(t1)
            t2 = ' '.join(t2)
            
            data = InputExample(texts=[t1, t2], label=simi)
            train_examples.append(data)


with codecs.open(opts.eval_file, 'r', 'utf-8') as f:
    t1 = None
    t2 = None
    simi = None
    for i, line in enumerate(tqdm.tqdm(f, file=f_log)):
        if i % 3 == 0:
            t1 = line.strip()
        elif i % 3 == 1:
            t2 = line.strip()
        elif i % 3 == 2:
            simi = float(line.strip())
            t1 = ' '.join(t1)
            t2 = ' '.join(t2)
            
            eval_s1.append(t1)
            eval_s2.append(t2)
            eval_score.append(simi)
            

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=opts.batch_size)
train_loss = losses.CosineSimilarityLoss(model)
evaluator = evaluation.EmbeddingSimilarityEvaluator(eval_s1, eval_s2, eval_score, main_similarity=SimilarityFunction.COSINE)
#device = torch.device("cuda:0")
#torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
#model = nn.parallel.DistributedDataParallel(model.cuda())
#model = BalancedDataParallel(8, model)
model = model.cuda()

def callback(score, epoch, steps):
    print(f'evaluation at {steps}/{epoch}:', score, file = f_log, flush=True)

print('start training', file = f_log, flush=True)
model.fit(train_objectives=[(train_dataloader, train_loss)],  
    epochs=opts.epoch,
    evaluator = evaluator, evaluation_steps=1000,  log_loss_steps = 1000,
    callback = callback,
    output_path = opts.output_dir,
    show_progress_bar=True,
    warmup_steps=100 )