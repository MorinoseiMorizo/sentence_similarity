import chainer


class BatchIterator(chainer.dataset.Iterator):
    def __init__(self, source_corpus, target_corpus, batch_size, repeat=True, reverse=True):
        assert len(source_corpus) == len(target_corpus)
        self.length = len(source_corpus)
        self.source_corpus = source_corpus
        self.target_corpus = target_corpus
        self.data_size = len(source_corpus)
        self.batch_size = batch_size
        self.epoch = 0
        self.is_new_epoch = False
        self.repeat = repeat
        self.reverse = reverse
        self.offsets = [i * self.length // batch_size for i in range(batch_size)]
        self.max_iteration = self.length // batch_size
        self.iteration = 0

    def fill_batch(self, samples, reverse=True):
        bos_symbol = 1  # means beginning of the sentence <s>
        eos_symbol = 2  # 2 is a id of </s>
        batch_size = len(samples)
        max_len = max(len(x) for x in samples)
        batch = [[eos_symbol] * (max_len+2) for _ in range(batch_size)]  # +2 is for <s>, and </s>
        if reverse:
            for i, sentence in enumerate(samples):
                batch[i][max_len+1] = bos_symbol
                for j, word in enumerate(sentence):
                    batch[i][max_len-j] = int(word)

        else:
            for i, sentence in enumerate(samples):
                batch[i][0] = bos_symbol
                for j, word in enumerate(sentence):
                    batch[i][j+1] = int(word)

        return batch

    def __next__(self):
        if not self.repeat and self.iteration * self.batch_size >= self.length:
            raise StopIteration
        source_samples = self.fill_batch(
                    self.get_samples(self.source_corpus),
                    reverse=self.reverse
                )
        target_samples = self.fill_batch(
                    self.get_samples(self.target_corpus),
                    reverse=False
                )
        self.iteration += 1

        epoch = self.iteration * self.batch_size // self.length
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch

        # return list(zip(source_samples, target_samples))
        return source_samples, target_samples

    @property
    def epoch_detail(self):
        return self.iteration * self.batch_size / self.data_size

    def get_samples(self, dataset):
    #    return [dataset[(offset + self.iteration) % len(dataset)]
    #            for offset in self.offsets]
        it = self.iteration % self.max_iteration
        return dataset[(self.batch_size*it):(self.batch_size*(1+it))]

    def serialize(self, serializer):
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)


if __name__ == "__main__":
    f_lines = []
    with open("./data/id/train.ja", 'r') as f:
        for line in f:
            f_lines.append(line.rstrip("\n").split())

    e_lines = []
    with open("./data/id/train.en", 'r') as e:
        for line in e:
            e_lines.append(line.rstrip("\n").split())

    train_iter = BatchIterator(f_lines, e_lines, 20)
    for batch in train_iter:
        for source, target in batch:
            print(" ".join(str(x) for x in source))

    # val_iter = ParallelSequentialIterator(val, 1, repeat=False)
