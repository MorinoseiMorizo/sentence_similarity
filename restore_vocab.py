from argparse import ArgumentParser
from collections import defaultdict


def main():
    p = ArgumentParser("Restore sentences from word ids.")
    p.add_argument(
            '--input',
            type=str, metavar="FILE", required=True, help="source word ids file")
    p.add_argument(
            '--output',
            type=str, metavar="FILE", required=True, help="converted sentence file")
    p.add_argument(
            '--vocab',
            type=str, metavar="FILE", required=True, help="vocab file")
    args = p.parse_args()

    with open(args.vocab, 'r') as f:
        vocab = defaultdict(lambda: "<UNK>")
        for line in f:
            word_id, word = line.rstrip("\n").split("\t")
            vocab[word_id] = word

    with open(args.input, 'r') as input_file, open(args.output, 'w') as output_file:
        for input_line in input_file:
            words = [vocab[x] for x in input_line.rstrip("\n").split()]
            print(" ".join(words), file=output_file)


if __name__ == "__main__":
    main()
