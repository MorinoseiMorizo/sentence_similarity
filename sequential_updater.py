from chainer import training

class SequentialUpdater(training.StandardUpdater):

    def __init__(self, train_iter, optimizer, device):
        super(SequentialUpdater, self).__init__(
            train_iter, optimizer, device=device)

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')
        #report({"lr":optimizer.lr})

        # Get the next batch
        x, t = train_iter.__next__()

        # Compute the loss at this time step and accumulate it
        loss = optimizer.target(x, t)
        # optimizer.target.cleargrads()  # Clear the parameter gradients
        optimizer.target.zerograds()  # Clear the parameter gradients
        loss.backward()  # Backprop
        optimizer.update()  # Update the parameters
