from collections import defaultdict


class Vocabulary:
    vocab = ["<UNK>", "<s>", "</s>"]

    def __init__(self, sentences, vocab_size=10000):
        words = defaultdict(int)

        for line in sentences:
            for word in line.rstrip("\n").split():
                words[word] += 1

        for key, value in sorted(words.items(), key=lambda x: x[1], reverse=True)[:vocab_size-3]:
            self.vocab.append(key)

        self.vocab_size = len(self.vocab)

    def stoi(self, word):
        if word in self.vocab:
            return self.vocab.index(word)

        return 0

    def itos(self, word_id):
        return self.vocab[word_id]

    def sentence_to_ids(self, sentence, with_bos=True, with_eos=True):
        ids = [1] if with_bos else []
        for word in sentence.rstrip("\n").split():
            ids.append(self.stoi(word))

        if with_eos:
            ids.append(2)

        return ids

    def ids_to_sentence(self, ids):
        words = []
        for i in ids:
            words.append(self.itos(i))

        return words


if __name__ == "__main__":
    lines = open("./data/train.en", 'r').readlines()
    vocab = Vocabulary(lines)

    ids = vocab.sentence_to_ids("this is a pen .")
    print(ids)
    print(vocab.ids_to_sentence(ids))

    print(vocab.vocab_size)

    lines_id = [vocab.sentence_to_ids(x) for x in lines]
