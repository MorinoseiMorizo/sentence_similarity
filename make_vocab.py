from vocabulary import Vocabulary
from argparse import ArgumentParser


def main():
    p = ArgumentParser("Make vocabulary file from the corpus.")
    p.add_argument(
            '--input',
            type=str, metavar="FILE", required=True, help="source corpus file")
    p.add_argument(
            '--output',
            type=str, metavar="FILE", required=True, help="converted corpus file")
    p.add_argument(
            '--size',
            type=int, metavar="N", required=True, help="vocabulary size")
    args = p.parse_args()

    with open(args.input, 'r') as f:
        vocab = Vocabulary(f.readlines(), vocab_size=args.size)

    with open(args.output, 'w') as f:
        for i, word in enumerate(vocab.vocab):
            f.write(str(i) + "\t" + word + "\n")


if __name__ == "__main__":
    main()

