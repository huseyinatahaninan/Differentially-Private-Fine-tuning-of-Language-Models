import os
from privacy_tools import get_sigma
import argparse

parser = argparse.ArgumentParser(description='Fine-tuning BERT with differentially private compactor')

parser.add_argument('--task', default='SST-2', type=str , choices=['MNLI', 'QNLI', 'QQP', 'SST-2'], help='name of the downstream task')
parser.add_argument('--gpu_id', default=0, type=int, help='which GPU to use, current implementation only supports using a single GPU')
parser.add_argument('--to_console', action='store_true', help='output to console, for debug use')
parser.add_argument('--sess', type=str, default='default', help='session name')

#normal hyperparameters
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--batch_size', default=2000, type=int, help='batch size')
parser.add_argument('--epoch', default=50, type=int, help='number of epochs')
parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay')
parser.add_argument('--arch', default='roberta.base', type=str, choices=['roberta.base', 'roberta.large'], help='model architecture')
# 0.999 is the value used in the original BERT paper
parser.add_argument('--adam_beta2', default=0.999, type=float, help='second beta value of adam')

parser.add_argument('--max_sentences', default=50, type=int, help='max sentences per step. Use a smaller value if your GPU runs out of memory')
parser.add_argument('--max_tokens', default=8000, type=int, help='max tokens per step. Use a smaller value if your GPU runs out of memory')

#new hyperparameters
parser.add_argument('--eps', default=8, type=float, help='DP parameter epsilon')
parser.add_argument('--delta', default=1e-5, type=float, help='DP parameter delta')
parser.add_argument('--clip', default=10., type=float, help='clipping threshold of individual gradients')
parser.add_argument('--k', default=16, type=int, help='number of bottleneck dimension')
parser.add_argument('--accountant', default='prv', type=str, choices=['moments', 'prv'], help='privacy accounting method')


parser.add_argument('--fp32', action='store_true', help='use full precision or not')

args = parser.parse_args()

assert args.task in ['MNLI', 'QNLI', 'QQP', 'SST-2']


data_dir = '../glue_data/%s-bin'%args.task
output_dir = 'log_dir'
ckpt_dir = '../%s/model.pt'%args.arch



assert args.batch_size % args.max_sentences == 0
update_freq = args.batch_size // args.max_sentences

dataset_size_dict ={'MNLI':392702, 'QQP':363849, 'QNLI':104743, 'SST-2':67349}
dataset_size = dataset_size_dict[args.task]

if(args.eps > 0):
    q = args.batch_size/dataset_size
    steps = args.epoch * (dataset_size//args.batch_size)
    sigma, eps = get_sigma(q, steps, args.eps, args.delta, mode=args.accountant)
    if(args.accountant == 'moments'):
        from prv_accountant import Accountant
        accountant = Accountant(
            noise_multiplier=sigma,
            sampling_probability=q,
            delta=args.delta,
            eps_error=0.1,
            max_compositions=steps)       
        eps_low, eps_estimate, eps_upper = accountant.compute_epsilon(num_compositions=steps)
        prv_eps = eps_upper

    print('Noise standard deviation:', sigma, 'Epsilon: ', eps, 'Delta: ', args.delta)
    if(args.accountant == 'moments'):
        print('Epsilon will be %.3f if using PRV accountant.'%prv_eps)
else:
    sigma = -1
    eps = -1

output_cmd = ' >> '
if(args.to_console):
    output_cmd = ' 2>&1 | tee '

sess = args.sess

os.system('mkdir -p %s/%s'%(output_dir, args.task))


apdx = ' '
# new added hyparameters
apdx += ' --sigma %f --clip %f --k %d '%(sigma, args.clip, args.k)
metric='accuracy'
n_classes=2
if(args.task == 'MNLI'):
    n_classes = 3
    apdx += ' --valid-subset valid,valid1 '

if('base' in args.arch):
    args.arch = 'roberta_base'
else:
    args.arch = 'roberta_large'


if(not args.fp32):
    apdx += ' --fp16  --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 '

cmd = 'CUDA_VISIBLE_DEVICES=%d python train.py %s --save-dir %s  \
        --restore-file %s \
        --max-positions 512\
        --update-freq %d \
        --max-sentences %d --max-tokens %d \
        --task sentence_prediction \
        --reset-optimizer --reset-dataloader --reset-meters \
        --required-batch-size-multiple 1 \
        --init-token 0 --separator-token 2 \
        --arch %s \
        --criterion sentence_prediction %s \
        --num-classes %d \
        --dropout 0.1 --attention-dropout 0.1 \
        --weight-decay %f --optimizer adam --adam-betas "(0.9,%f)" --adam-eps 1e-06 \
        --clip-norm 0 --validate-interval-updates 1 \
        --lr-scheduler polynomial_decay --lr %f --warmup-ratio 0.06 --sess %s \
        --max-epoch %d --seed %d  --no-progress-bar --log-interval 100 --no-epoch-checkpoints --no-last-checkpoints --no-best-checkpoints \
        --find-unused-parameters --skip-invalid-size-inputs-valid-test --truncate-sequence --embedding-normalize  \
        --tensorboard-logdir . --bert-pooler --pooler-dropout 0.1 \
        --best-checkpoint-metric %s --maximize-best-checkpoint-metric %s %s/%s/%s_train_log.txt'%(args.gpu_id, data_dir, output_dir, ckpt_dir, 
            update_freq, args.max_sentences, args.max_tokens, args.arch, apdx, n_classes, args.weight_decay, args.adam_beta2, args.lr, args.sess, args.epoch, args.seed, 
            metric, output_cmd, output_dir, args.task, sess)

os.system(cmd)