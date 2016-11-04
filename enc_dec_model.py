import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import report

class EncDecModel(chainer.Chain):

    def __init__(self, N_SOURCE_VOCAB, N_TARGET_VOCAB, N_EMBED, N_HIDDEN, train=True):
        super(EncDecModel, self).__init__(
                # Encoder
                enc_embed=L.EmbedID(N_SOURCE_VOCAB, N_EMBED),
                enc_lstm_1=L.LSTM(N_EMBED, N_HIDDEN),
                enc_lstm_2=L.LSTM(N_HIDDEN, N_HIDDEN),
                # Decoder initializer
                enc_dec_1_c=L.Linear(N_HIDDEN, N_HIDDEN),
                enc_dec_1_h=L.Linear(N_HIDDEN, N_HIDDEN),
                enc_dec_2_c=L.Linear(N_HIDDEN, N_HIDDEN),
                enc_dec_2_h=L.Linear(N_HIDDEN, N_HIDDEN),
                # Decoder
                dec_embed=L.EmbedID(N_TARGET_VOCAB, N_EMBED),
                dec_lstm_1=L.LSTM(N_EMBED, N_HIDDEN),
                dec_lstm_2=L.LSTM(N_HIDDEN, N_HIDDEN),
                dec_output=L.Linear(N_HIDDEN, N_TARGET_VOCAB),
        )
        for param in self.params():
            param.data[...] = self.xp.random.uniform(-0.08, 0.08, param.data.shape)
        self.train = train
        self.src_vocab_size = N_SOURCE_VOCAB
        self.trg_vocab_size = N_TARGET_VOCAB
        self.embed_size = N_EMBED
        self.hidden_size = N_HIDDEN

    def reset_state(self):
        self.enc_lstm_1.reset_state()
        self.enc_lstm_2.reset_state()
        self.dec_lstm_1.reset_state()
        self.dec_lstm_2.reset_state()

    def _encode(self, src_samples):
        src_samples_array = chainer.Variable(self.xp.array(src_samples, dtype=self.xp.int32).T)
        for words in src_samples_array:
            e_h0 = self.enc_embed(words)
            e_h1 = self.enc_lstm_1(e_h0)
            e_h2 = self.enc_lstm_2(e_h1)
        return e_h2

    def _init_decoder(self, e_h1, e_c1, e_h2, e_c2):
        d_h1 = self.enc_dec_1_h(e_h1)
        d_c1 = self.enc_dec_1_c(e_c1)
        d_h2 = self.enc_dec_2_h(e_h2)
        d_c2 = self.enc_dec_2_c(e_c2)
        self.dec_lstm_1.set_state(d_c1, d_h1)
        self.dec_lstm_2.set_state(d_c2, d_h2)

    def _decode(self, trg_samples):
        trg_samples_array = chainer.Variable(self.xp.array(trg_samples, dtype=self.xp.int32).T)
        loss = chainer.Variable(self.xp.zeros((), dtype=self.xp.float32))
        num_words = len(trg_samples_array)-1
        for y, t in zip(trg_samples_array, trg_samples_array[1:]):
            d_h0 = self.dec_embed(y)
            d_h1 = self.dec_lstm_1(d_h0)
            d_h2 = self.dec_lstm_2(d_h1)
            z = self.dec_output(d_h2)
            loss += self._loss(z, t)
        return loss, num_words

    def _loss(self, y, t):
        return F.softmax_cross_entropy(y, t)

    def __call__(self, src_samples, trg_samples):
        self.reset_state()
        self._encode(src_samples)
        self._init_decoder(
                self.enc_lstm_1.h, self.enc_lstm_1.c,
                self.enc_lstm_2.h, self.enc_lstm_2.c,
                )
        loss, num_words = self._decode(trg_samples)
        report({'loss': loss}, self)
        report({'num_words': num_words}, self)
        return loss

    def _test_decode(self, batch_size, bos_id, eos_id, limit):
        z_list = []
        y = [bos_id for _ in range(batch_size)]
        y = chainer.Variable(self.xp.array(y, dtype=self.xp.int32))
        for i in range(limit):
            d_h0 = self.dec_embed(y)
            d_h1 = self.dec_lstm_1(d_h0)
            d_h2 = self.dec_lstm_2(d_h1)
            z = self.dec_output(d_h2)
            z = [int(w) for w in z.data.argmax(1)]
            if all(w == eos_id for w in z):
                break
            z_list.append(z)
            y = chainer.Variable(self.xp.array(z, dtype=self.xp.int32))
        return z_list

    def forward_test(self, src_samples, trg_samples):
        batch_size = len(src_samples)
        bos_id = 1
        eos_id = 2
        limit = 60

        self.reset_state()
        self._encode(src_samples)
        self._init_decoder(
                self.enc_lstm_1.h, self.enc_lstm_1.c,
                self.enc_lstm_2.h, self.enc_lstm_2.c,
                )
        z = self._test_decode(batch_size, bos_id, eos_id, limit)
        ret = list(map(list, zip(*z)))

        return trg_samples, ret

    def forward_get_sentence_vector(self, src_samples, trg_samples):
        self.reset_state()
        sentence_vector = self._encode(src_samples)

        return sentence_vector
