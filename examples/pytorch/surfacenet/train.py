import torch
import sys
import os
import plotly
import numpy as np
import argparse
import torch.nn.functional as F
from torch.optim import Adam
import random
import time
import re
import gc
import glob
import tqdm
import functools
import subprocess
import multiprocessing
from models import LapDeepModel
import sampler
from sampler import sample_batch

random.seed(17)
# Training settings
parser = argparse.ArgumentParser(description='Normal Predictor')

# Model Setting
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--layer', type=int, default=15)

parser.add_argument('--no-cuda', action='store_true', default=False)

# Training Process Control
parser.add_argument('--data-size', default='10k')
parser.add_argument('--data-patch', default='1024')
parser.add_argument('--num-epoch', type=int, default=300, metavar='N',
                    help='num of training epochs (default: 100)')
parser.add_argument('--start-epoch', type=int, default=0)
parser.add_argument('--num-updates', type=int, default=1250, metavar='N',
                    help='num of training epochs (default: 1250')
parser.add_argument('--no-test', action="store_true")
parser.add_argument('--half-lr', type=int, default=20,
                    help='Halves lr every N epochs, -1 means no')
parser.add_argument('--only-forward-test', action="store_true",
                    help="Used to generate results")
parser.add_argument('--dump-dir', default='/dev/shm/')

parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units')

# Optimizing
parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate')
parser.add_argument('--optimizer', default='adam')
parser.add_argument('--dense', action='store_true')

# Experimental Options
parser.add_argument('--uniform-mesh', action="store_true",
                    help='Scale Mesh to uniform to train across categories')

parser.add_argument('--var-size', action="store_true",
                    help='Use variable size graphs')
parser.add_argument('--threshold', type=int, default=None,
                    help='Threshold for adaptive batch size')
parser.add_argument('--shuffle', action='store_true',
                    help='shuffle dataset')
parser.add_argument('--use-schedule', type=str, default=None)

parser.add_argument(
    '--additional-opt',
    default=['hack1'],
    action='append',
    choices=[
        'hack1',
        'intrinsic',
        'amsgrad',
        ''])

parser.add_argument('--debug', action='store_true', help='Not writing to file')
parser.add_argument('--pre-load', action='store_true',
                    help='Offload computation')
parser.set_defaults(uniform_mesh=True)


def main():
    args = parser.parse_args()
    if args.var_size and args.use_schedule is None:
        assert(args.threshold is not None)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.var_size:
        args.data_path = f'{os.environ["HOME"]}/{args.data_size}-var/'
        args.test_path = '@'
    else:
        args.data_path = f'{os.environ["HOME"]}/{args.data_size}/' \
            f'train/{args.data_patch}/'
        args.test_path = f'{os.environ["HOME"]}/{args.data_size}/' \
            f'test/{args.data_patch}/'
    args.result_prefix = f'{args.data_size}_{args.data_patch}_{args.lr}'
    device = torch.device('cuda' if args.cuda else 'cpu')
    vis = None
    result_identifier = args.result_prefix

    def custom_logging(stuff):
        if args.debug:
            # also to err
            print(f'{result_identifier}::{stuff}', file=sys.stderr)
        else:
            print(f'{result_identifier}::{stuff}')  # also to out
            logfile = f'log/{result_identifier}.log'
            with open(logfile, 'a') as fp:
                print(stuff, file=fp)

    custom_logging(args)
    custom_logging(subprocess.check_output('hostname'))
    custom_logging(subprocess.check_output(
        'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader',
        shell=True))

    def loss_fun(inputs, mask, targets, **kwargs):
        inputs = F.normalize(inputs, p=2, dim=1)
        inner = torch.sum(inputs * targets, dim=1)
        loss = 1 - inner ** 2
        return torch.mean(torch.masked_select(loss, mask.squeeze().bool()))

    def mean_angle_deviation(inputs, mask, targets, **kwargs):
        inputs = F.normalize(inputs, p=2, dim=1)
        inner = torch.sum(inputs * targets, dim=1)
        inner = torch.clamp(torch.abs(inner), 0, 1)
        loss = torch.acos(inner)
        return torch.mean(torch.masked_select(loss, mask.squeeze().bool()))

    input_type_to_dim = {'V': 3, 'G': 1, 'wks': 100, 'cor_V': 3,
                         'N': 3, 'curv4': 4}
    args.input_dim = input_type_to_dim['V']
    args.output_dim = 3

    bnmode = ''  # normal bn
    if 'groupnorm' in args.additional_opt:
        bnmode = 'group'
    if 'nobn' in args.additional_opt:
        bnmode = None

    args.bottleneck = False
    lap_opts = {'layers': args.layer,
                'bnmode': bnmode,
                'only_lap': 'only_lap' in args.additional_opt,
                'nofirstId': False,
                'num_hidden': args.hidden}
    model = LapDeepModel(args.input_dim, args.output_dim, **lap_opts)

    model.to(device)

    custom_logging("Num parameters {}".format(
        sum(p.numel() for p in model.parameters())))

    early_optimizer = Adam(model.parameters(), lr=args.lr,
                           amsgrad='amsgrad' in args.additional_opt)

    if not args.only_forward_test:
        seq_names = sorted(glob.glob(args.data_path + '**/*.obj',
                                     recursive=True))
    else:
        seq_names = []
    custom_logging(f'SEQ:{len(seq_names)}')

    if args.test_path != '@':
        train_seq_names = seq_names
        test_seq_names = sorted(glob.glob(f'{args.test_path}/**/*.obj',
                                          recursive=True))
    else:
        if args.no_test:
            train_seq_names = seq_names
        else:
            # 80/20 seperation
            sep_length = len(seq_names)//10*8
            if args.shuffle:
                print("Shuffle dataset")
                random.shuffle(seq_names)
            train_seq_names = seq_names[:sep_length]
            test_seq_names = seq_names[sep_length:]

    real_epoch_counter = 0

    if not args.var_size and not args.shuffle:
        train_seq_names = train_seq_names[:args.batch_size * args.num_updates]

    if args.use_schedule:
        schedule = sampler.load_schedule(args.use_schedule)
        schedule = schedule[:args.num_updates]
        train_seq_names = train_seq_names[:schedule[-1][-1] + 1]

    if args.pre_load:
        print("start to preload")
        # pickle-able for imap
        readnpz = functools.partial(sampler.read_npz, args=args)
        # https://github.com/pytorch/pytorch/issues/973
        torch.multiprocessing.set_sharing_strategy('file_system')
        if not args.only_forward_test:
            if True:
                #train_seq_names = train_seq_names[:16]
                train_seq_names = [readnpz(t)
                                   for t in tqdm.tqdm(train_seq_names,
                                                      ncols=0)]
            else:
                with torch.multiprocessing.Pool(1) as p:
                    work = p.imap(readnpz, train_seq_names)
                    train_seq_names = tqdm.tqdm(work,
                                                total=len(train_seq_names),
                                                ncols=0)
                    train_seq_names = list(train_seq_names)
            train_seq_names = [t for t in train_seq_names if t is not None]

        if not args.no_test:
            test_seq_names = [readnpz(t)
                              for t in tqdm.tqdm(test_seq_names, ncols=0)]
            test_seq_names = [t for t in test_seq_names if t is not None]
        else:
            test_seq_names = []
        print('Train size:', len(train_seq_names),
              ' Test size:', len(test_seq_names))
        print("finish preload")

    train_loss = []
    test_loss = []

    for epoch in range(args.start_epoch, args.num_epoch):
        if not args.only_forward_test:
            model.train()
            loss_value = 0
            mad = 0
            # Train
            if args.use_schedule is None:
                nbatch = args.num_updates
                loader = sample_batch(train_seq_names, args, nbatch)
            else:
                nbatch = len(schedule)
                loader = sampler.produce_batch_from_schedule(schedule, train_seq_names)
            pb = tqdm.tqdm(loader, total=nbatch, ncols=80)
            #torch.cuda.synchronize()
            #t0 = time.time()
            for num_up, (graph, curr_name) in enumerate(pb):
                #torch.cuda.synchronize()
                #t1 = time.time()
                inputs = graph.ndata.pop('input')
                outputs = model(graph, inputs)
                if (torch.isnan(outputs.detach())).any():
                    assert False, f'NANNNN {curr_name[0]} outputs'

                targets = graph.ndata.pop('target')
                mask = graph.ndata['mask']
                loss = loss_fun(outputs, mask, targets)
                mad += mean_angle_deviation(outputs, mask, targets).item()
                #torch.cuda.synchronize()
                #t2 = time.time()

                early_optimizer.zero_grad()
                loss.backward()
                early_optimizer.step()
                loss_value += loss.item()
                if np.isnan(loss_value):
                    assert False, f'NANNNN {curr_name[0]} LOSS'
                pb.set_postfix(loss=loss_value / (num_up + 1),
                               mad=mad / (num_up + 1))
                #torch.cuda.synchronize()
                #t3 = time.time()
                #print(graph.batch_size, graph.number_of_nodes(), t3 - t0)
                #t0 = t3

            custom_logging("Train {}, loss {}, mad {}, time {}"
                           .format(epoch, loss_value / args.num_updates,
                                   mad / args.num_updates,
                                   pb.last_print_t - pb.start_t))
            train_loss.append(loss_value / args.num_updates)

        # Evaluate
        with torch.no_grad():
            loss_value = 0
            mad = 0
            if not args.no_test:
                test_trials = np.ceil(len(test_seq_names) // args.batch_size)
                for graph, names in tqdm.tqdm(sample_batch(test_seq_names,
                                                           args, test_trials),
                                              total=test_trials):
                    inputs = graph.ndata.pop('input')
                    outputs = model(graph, inputs)
                    if (torch.isnan(outputs.detach())).any():
                        assert False, f'NANNNN {curr_name[0]} outputs'

                    targets = graph.ndata.pop('target')
                    mask = graph.ndata['mask']
                    outputs = outputs
                    loss = loss_fun(outputs, mask, targets)

                    loss_value += loss.item()
                    mad += mean_angle_deviation(outputs, mask, targets).item()

                    if args.only_forward_test:
                        directory = f'{args.dump_dir}/{args.result_prefix}/'
                        if not os.path.exists(directory):
                            os.makedirs(directory)
                        for name, targ in zip(names, outputs):
                            np.savetxt(directory + os.path.basename(name) +
                                       '.csv', targ.cpu().numpy(),
                                       delimiter=',')

                custom_logging("Eval {}, loss {}, mad {}".format(
                    epoch, loss_value / test_trials, mad/test_trials))
                test_loss.append(loss_value / test_trials)
            sys.stdout.flush()
        if args.only_forward_test:
            return
        if epoch % 10 == 9 and not args.debug:
            torch.save({'weights': model.state_dict(),
                        'optimizer': early_optimizer.state_dict(),
                        'epoch': epoch},
                       'pts/' + args.result_prefix + '_normal_state.pts')

        if args.half_lr > 0 and epoch > 100 and epoch % args.half_lr == 0:
            for param_group in early_optimizer.param_groups:
                param_group['lr'] *= 0.5
            custom_logging(f'Halving LR to {param_group["lr"]}')

    torch.save(model.state_dict(),
               'pts/' + args.result_prefix + '_normal_state.pts')


if __name__ == '__main__':
    main()
