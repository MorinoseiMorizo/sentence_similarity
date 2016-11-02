import numpy as np
import chainer
from chainer import training
from chainer.training import extensions
from batch_iterator import BatchIterator
from sequential_updater import SequentialUpdater
from mt_evaluator import MT_Evaluator
from enc_dec_model import EncDecModel
from attentional_model import AttentionalModel
import argparse

# Routine to rewrite the result dictionary of LogReport to add perplexity
# values
def compute_perplexity(result):
    result['perplexity'] = np.exp(result['main/loss']) / result['main/num_words']
    if 'validation/main/loss' in result:
        result['val_perplexity'] = np.exp(result['validation/main/loss'])

class adjust_learning_rate(training.Extension):
    def __call__(self, trainer):
        epoch = trainer.updater.epoch
        if epoch >= 5:
            optimizer = trainer.updater.get_optimizer('main')
            optimizer.lr = optimizer.lr / 2.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, required=True,
                        help="Training model. \
                              \'encdec\' for Encoder Decoder Model, \
                              \'attention\' for Attentional Enc-Dec Model.")
    parser.add_argument('--source', type=str, required=True,
                        help='Source corpus')
    parser.add_argument('--target', type=str, required=True,
                        help='Target corpus')
    parser.add_argument('--source_test', type=str, required=True,
                        help='Source corpus for test')
    parser.add_argument('--target_test', type=str, required=True,
                        help='Target corpus for test')
    parser.add_argument('--srcvocab', type=int, default=10000,
                        help='Size of the source language vocabulary')
    parser.add_argument('--trgvocab', type=int, default=10000,
                        help='Size of the target language vocabulary')
    parser.add_argument('--embed', type=int, default=1024,
                        help='Size of the embed layer')
    parser.add_argument('--hidden', type=int, default=1024,
                        help='Size of the each hidden layer')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--test_batchsize', type=int, default=100,
                        help='Number of examples in each mini-batch for testing')
    parser.add_argument('--epoch', '-e', type=int, default=39,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gradclip', '-c', type=float, default=5,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--test', type=bool, default=False,
                        help='If True, train with a smaller dataset')
    parser.set_defaults(test=False)
    args = parser.parse_args()

    if args.model not in ["encdec", "attentional"]:
        print("specify --model option correctly")
        return -1

    # make an iterator for train
    f_lines = []
    with open(args.source, 'r') as f:
        for line in f:
            f_lines.append(line.rstrip("\n").split())

    e_lines = []
    with open(args.target, 'r') as e:
        for line in e:
            e_lines.append(line.rstrip("\n").split())

    if args.test:
        f_lines = f_lines[:1000]
        e_lines = e_lines[:1000]

    train_iter = BatchIterator(f_lines, e_lines, args.batchsize)

    # make an iterator for test
    test_f_lines = []
    with open(args.source_test, 'r') as f:
        for line in f:
            test_f_lines.append(line.rstrip("\n").split())

    test_e_lines = []
    with open(args.target_test, 'r') as e:
        for line in e:
            test_e_lines.append(line.rstrip("\n").split())

    test_iter = BatchIterator(test_f_lines, test_e_lines, args.test_batchsize, repeat=False)

    if args.model == "encdec":
        model = EncDecModel(args.srcvocab, args.trgvocab, args.embed, args.hidden)
    if args.model == "attentional":
        model = AttentionalModel(args.srcvocab, args.trgvocab, args.embed, args.hidden)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # make the GPU current
        model.to_gpu()

    # Set up an optimizer
    optimizer = chainer.optimizers.Adam(alpha=0.001)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))
    #optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    # Set up a trainer
    updater = SequentialUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'))

    # trainer extensions
    interval = 10
    trainer.extend(extensions.LogReport(postprocess=compute_perplexity,
                                        trigger=(interval, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'perplexity']
    ), trigger=(interval, 'iteration'))
    trainer.extend(MT_Evaluator(test_iter, model, device=args.gpu, eval_func=model.forward_test, out_dir=args.out), trigger=(1, 'epoch'))
    trainer.extend(extensions.ProgressBar(update_interval=1))
    #trainer.extend(extensions.snapshot_object(
    #    model, 'model_epoch_{.updater.epoch}'))
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

if __name__ == "__main__":
    main()
