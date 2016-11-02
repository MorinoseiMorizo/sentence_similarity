#!/usr/bin/env python3
import math
from argparse import ArgumentParser
from collections import defaultdict


def get_bleu_stats(ref, hyp, N=4):
    ref = list(map(str, ref))
    hyp = list(map(str, hyp))
    stats = defaultdict(int, {'rl': len(ref), 'hl': len(hyp)})
    N = len(hyp) if len(hyp) < N else N
    for n in range(N): 
        matched = 0
        possible = defaultdict(int)
        for k in range(len(ref) - n):
            possible[tuple(ref[k : k + n + 1])] += 1
        for k in range(len(hyp) - n):
            ngram = tuple(hyp[k : k + n + 1])
            if possible[ngram] > 0:
                possible[ngram] -= 1
                matched += 1
        stats['d' + str(n + 1)] = len(hyp) - n
        stats['n' + str(n + 1)] = matched
    return stats


def calculate_bleu(stats, N=4):
    np = 0.0
    zero_flag = False
    for n in range(N):
        nn = stats['n' + str(n + 1)]
        if nn == 0:
            zero_flag = True
            continue
        dd = stats['d' + str(n + 1)]
        np += math.log(nn) - math.log(dd)
    bp = 1.0 - stats['rl'] / stats['hl']
    if bp > 0.0:
        bp = 0.0
    ret = {'rl': stats['rl'], 'hl': stats['hl'], 'bp': math.exp(bp),
           'ratio': stats['hl'] / stats['rl']}
    if zero_flag is False:
        for n in range(N):
            ret[n+1] = stats['n' + str(n+1)] / stats['d' + str(n+1)]
        ret['bleu'] = math.exp(np / N + bp)
    else:
        for n in range(N):
            ret[n+1] = 0.0
        ret['bleu'] = 0.0
    return ret


def parse_args():
    p = ArgumentParser()
    p.add_argument(
            '--ref',
            type=str, metavar='FILE', required=True,
            help='reference corpus')
    p.add_argument(
            '--hyp',
            type=str, metavar='FILE', required=True,
            help='hypothesis corpus')
    p.add_argument(
            '--full',
            type=bool, metavar='True/False', default=True,
            help='Show full analysis or BLEU only')
    return p.parse_args()


def get_bleu_corpus(refs, hyps, input_is_str=False):
    stats = defaultdict(int)
    for ref, hyp in zip(refs, hyps):
        if input_is_str:
            ref = ref.split()
            hyp = hyp.split()
        for k, v in get_bleu_stats(ref, hyp).items():
            stats[k] += v
    ret = calculate_bleu(stats)
    return ret


def main():
    args = parse_args()

    with open(args.ref, 'r') as refs, open(args.hyp, 'r') as hyps:
        bleu = get_bleu_corpus(refs.readlines(), hyps.readlines(), input_is_str=True)
        print("BLEU = %.6f, %.6f/%.6f/%.6f/%.6f (BP=%.6f, ratio=%.6f, hyp_len=%d, ref_len=%d)" %
              (bleu["bleu"], bleu[1], bleu[2], bleu[3], bleu[4], bleu["bp"], bleu["ratio"], bleu["hl"], bleu["rl"]))


if __name__ == '__main__':
    main()
