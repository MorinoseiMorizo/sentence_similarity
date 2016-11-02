import copy

import six

from chainer.training import extensions
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import link
from chainer import reporter as reporter_module
import bleu
import os


class MT_Evaluator(extensions.Evaluator):
    def __init__(self, iterator, target, converter=convert.concat_examples, out_dir="./out",
                 device=None, eval_hook=None, eval_func=None):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator

        if isinstance(target, link.Link):
            target = {'main': target}
        self._targets = target

        self.converter = converter
        self.device = device
        self.eval_hook = eval_hook
        self.eval_func = eval_func
        self.out_dir = out_dir
        self.epoch = 1

        try:
            os.makedirs(self.out_dir)
        except:
            pass

    def __call__(self, trainer=None):
        """Executes the evaluator extension.

        Unlike usual extensions, this extension can be executed without passing
        a trainer object. This extension reports the performance on validation
        dataset using the :func:`~chainer.report` function. Thus, users can use
        this extension independently from any trainer by manually configuring
        a :class:`~chainer.Reporter` object.

        Args:
            trainer (~chainer.training.Trainer): Trainer object that invokes
                this extension. It can be omitted in case of calling this
                extension manually.

        Returns:
            dict: Result dictionary that contains mean statistics of values
                reported by the evaluation function.

        """
        # set up a reporter
        reporter = reporter_module.Reporter()
        if hasattr(self, 'name'):
            prefix = self.name + '/'
        else:
            prefix = ''
        for name, target in six.iteritems(self._targets):
            reporter.add_observer(prefix + name, target)
            reporter.add_observers(prefix + name,
                                   target.namedlinks(skipself=True))

        with reporter:
            result = self.evaluate()

        reporter_module.report(result)
        return result

    def evaluate(self):
        """Evaluates the model and returns a result dictionary.

        This method runs the evaluation loop over the validation dataset. It
        accumulates the reported values to :class:`~chainer.DictSummary` and
        returns a dictionary whose values are means computed by the summary.

        Users can override this method to customize the evaluation routine.

        Returns:
            dict: Result dictionary. This dictionary is further reported via
                :func:`~chainer.report` without specifying any observer.

        """
        iterator = self._iterators['main']
        target = self._targets['main']
        eval_func = self.eval_func or target
        corpus_length = iterator.length

        if self.eval_hook:
            self.eval_hook(self)
        it = copy.copy(iterator)

        refs, hyps = [], []

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                src_sample, hyp_sample = batch
                ref, hyp = eval_func(src_sample, hyp_sample)

            for x, y in zip(ref, hyp):
                refs.append(x)
                hyps.append(list(map(int, y)))

        refs = refs[:corpus_length]
        hyps = hyps[:corpus_length]

        self.save_output(hyps, self.out_dir + "/" + str(self.epoch))
        self.epoch += 1

        return {'bleu': self.compute_bleu(refs, hyps)}

    def save_output(self, sentences, path):
        with open(path, 'w') as f:
            for x in sentences:
                x = " ".join(str(_) for _ in x)
                f.write(x + '\n')

    def compute_bleu(self, refs, hyps):
        return bleu.get_bleu_corpus(refs, hyps)['bleu']
