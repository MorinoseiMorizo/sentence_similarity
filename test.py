import chainer
from batch_iterator import BatchIterator
from enc_dec_model import EncDecModel
import argparse
import copy

def run_test(eval_func, test_iter, corpus_length):
    hyps = []
    it = copy.copy(test_iter)

    for batch in it:
        src_sample, hyp_sample = batch
        ref, hyp = eval_func(src_sample, hyp_sample)

        for x in hyp:
            hyps.append(x)

    hyps = hyps[:corpus_length]

    return hyps

def print_list_sentences(list_sentences):
    for line in list_sentences:
        print(" ".join(str(x) for x in line))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, required=True,
                        help='Model snapshot file to use')
    parser.add_argument('--source', type=str, required=True,
                        help='Source corpus')
    parser.add_argument('--srcvocab', type=int, default=10000,
                        help='Size of the source language vocabulary')
    parser.add_argument('--trgvocab', type=int, default=10000,
                        help='Size of the target language vocabulary')
    parser.add_argument('--embed', type=int, default=1024,
                        help='Size of the embed layer')
    parser.add_argument('--hidden', type=int, default=1024,
                        help='Size of the each hidden layer')
    parser.add_argument('--batchsize', type=int, default=1,
                        help='Batchsize for testing')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    # make an iterator for train
    f_lines = []
    e_lines = []
    with open(args.source, 'r') as f:
        for line in f:
            f_lines.append(line.rstrip("\n").split())
            e_lines.append("")

    test_iter = BatchIterator(f_lines, e_lines, args.batchsize, repeat=False)

    model = EncDecModel(args.srcvocab, args.trgvocab, args.embed, args.hidden)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # make the GPU current
        model.to_gpu()

    chainer.serializers.load_npz(args.model_file, model)

    hyps = run_test(model.forward_test, test_iter, len(f_lines))
    
    print_list_sentences(hyps)

if __name__ == "__main__":
    main()
