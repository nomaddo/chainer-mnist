import chainer
from chainer import training
from chainer import datasets
from chainer.training import extensions
from chainer import functions as F
from chainer import links as L
import numpy as np

device = 0


class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l0 = L.Convolution2D(None, 32, 3)
            self.l1 = L.Convolution2D(None, 64, 3)
            self.l2 = L.Convolution2D(None, 128, 3)
            self.fc0 = L.Linear(None, 512)
            self.fc1 = L.Linear(None, 10)

    def __call__(self, x):
        h0 = F.max_pooling_2d(F.relu(self.l0(x)), 2)
        h1 = F.max_pooling_2d(F.relu(self.l1(h0)), 2)
        h2 = F.max_pooling_2d(F.relu(self.l2(h1)), 2)
        return self.fc1(F.relu(self.fc0(h2)))


def main():
    train_full, test_full = chainer.datasets.get_mnist(ndim=3)
    train = datasets.SubDataset(train_full, 0, 1000)
    test = datasets.SubDataset(test_full, 0, 1000)
    model = L.Classifier(MLP(100, 10))

    batchsize = 100
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(
        test, batchsize, repeat=False, shuffle=False)

    opt = chainer.optimizers.Adam()
    opt.setup(model)

    updater = training.StandardUpdater(train_iter, opt, device=device)

    epoch = 20
    trainer = training.Trainer(updater, (epoch, 'epoch'), out='./result')

    snapshot_interval = (1, 'epoch')
    trainer.extend(extensions.Evaluator(test_iter, model,
                                        device=device), trigger=snapshot_interval)
    trainer.extend(extensions.ProgressBar(), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=snapshot_interval))
    trainer.extend(extensions.snapshot(
        filename='snapshot_epoch_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_epoch_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.PrintReport(
        ["epoch", "main/accuracy", 'validation/main/accuracy']), trigger=(1, 'epoch'))

    trainer.run()


if __name__ == '__main__':
    main()
