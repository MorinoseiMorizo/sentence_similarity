from argparse import ArgumentParser
from collections import defaultdict


def main():
    p = ArgumentParser("Make corpus file converted to word ids.")
    p.add_argument(
            '--input',
            type=str, metavar="FILE", required=True, help="source corpus file")
    p.add_argument(
            '--output',
            type=str, metavar="FILE", required=True, help="converted corpus file")
    p.add_argument(
            '--vocab',
            type=str, metavar="FILE", required=True, help="vocab file")
    args = p.parse_args()

    with open(args.vocab, 'r') as f:
        vocab = defaultdict(lambda: '0')  # <UNK> is 0.
        for line in f:
            word_id, word = line.rstrip("\n").split("\t")
            vocab[word] = word_id

    with open(args.input, 'r') as input_file, open(args.output, 'w') as output_file:
        for input_line in input_file:
            word_ids = [vocab[x] for x in input_line.rstrip("\n").split()]
            print(" ".join(word_ids), file=output_file)


if __name__ == "__main__":
    main()
