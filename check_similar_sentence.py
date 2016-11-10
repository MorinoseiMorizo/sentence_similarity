import chainer
from batch_iterator import BatchIterator
from enc_dec_model import EncDecModel
from vector_representation import get_sentence_vector
import numpy as np
import scipy.spatial.distance as dis
import argparse
import sys

def euclidean_distance(a, b):
    return np.linalg.norm(a-b)

def cosine_similarity(a, b):
    return dis.cosine(a, b)

def find_top_n_similar_sentences(training_sentence_vectors, test_sentence_vector, training_sentences):
    distances = []
    for sentence_vector, sentence in zip(training_sentence_vectors, training_sentences):
        distances.append((cosine_similarity(sentence_vector, test_sentence_vector), sentence))

    distances = sorted(distances, key=lambda x: x[0])
    
    for score, sentence in distances[:10]:
        cat_sentence = " ".join(sentence)
        print(str(score) + "\t" + cat_sentence)

def make_batch_then_get_sentence_vector(sentence, model):
    test_iter = BatchIterator([sentence], [""], 1, repeat=False)
    
    sentence_vector = get_sentence_vector(model.forward_get_sentence_vector, test_iter, 1)

    return sentence_vector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vector', type=str, required=True,
                        help='Sentence vectors file')
    parser.add_argument('--sentence', type=str, required=True,
                        help='Corpus file')
    parser.add_argument('--model_file', type=str, required=True,
                        help='Model snapshot file to use')
    parser.add_argument('--srcvocab', type=int, default=10000,
                        help='Size of the source language vocabulary')
    parser.add_argument('--trgvocab', type=int, default=10000,
                        help='Size of the target language vocabulary')
    parser.add_argument('--embed', type=int, default=1024,
                        help='Size of the embed layer')
    parser.add_argument('--hidden', type=int, default=1024,
                        help='Size of the each hidden layer')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    # loading model
    model = EncDecModel(args.srcvocab, args.trgvocab, args.embed, args.hidden)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # make the GPU current
        model.to_gpu()
    chainer.serializers.load_npz(args.model_file, model)

    # load training sentence vectors
    training_sentence_vectors = np.loadtxt(args.vector)

    # load training sentences
    training_sentences = []
    with open(args.sentence, 'r') as f:
        for line in f:
            training_sentences.append(line.rstrip("\n").split())

    for line in sys.stdin:
        test_sentence_vector = make_batch_then_get_sentence_vector(line.rstrip("\n").split(), model)
        find_top_n_similar_sentences(training_sentence_vectors, test_sentence_vector, training_sentences)

if __name__ == "__main__":
    main()
