import torch.nn as nn
import torch

def load_fr_julia(Nfiles):
    from julia.api import Julia
    jl = Julia(compiled_modules=False, debug=False)

    import julia
    from julia import Pkg
    from julia import Main

    import subprocess

    import numpy as np

    Pkg.activate('/export/nfs0home/pstaudt/projects/gal-learn/GalLearn')

    # Attempt to locate the Julia executable path
    try:
        julia_path = subprocess.check_output(
                ["which", "julia"]
            ).decode("utf-8").strip()
        print("Julia executable path:", julia_path)
        
        # Optional: Print Julia version to verify
        julia_version = subprocess.check_output(
                [julia_path, "--version"]
            ).decode("utf-8").strip()
        print("Julia version:", julia_version)
    except subprocess.CalledProcessError:
        print("Julia executable not found in PATH")

    Main.include('image_loader.jl')
    obs_sorted, ys, ys_sorted, X, files = julia.Main.image_loader.load_data(
        Nfiles=Nfiles,
        res=1500
    )

    ys = torch.FloatTensor(np.array(ys_sorted))
    X = torch.FloatTensor(X)

    return X, ys

class Net(nn.Module):
    def __init__(
                self,
                activation,
                kernel_size,
                N_conv1_out_chan,
                N_conv2_out_chan,
                N_out_channels 
            ):

        super(Net, self).__init__()

        self.activation = activation
        self.N_out_channels = N_out_channels
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=N_conv1_out_chan,
            kernel_size=kernel_size
        )
        self.conv2 = nn.Conv2d(
            in_channels=N_conv1_out_chan,
            out_channels=N_conv2_out_chan,
            kernel_size=kernel_size
        )
        #self.conv3 = nn.Conv2d(
        #    in_channels=3,
        #    out_channels=1,
        #    kernel_size=kernel_size
        #)
        #self.conv4 = nn.Conv2d(
        #    in_channels=1,
        #    out_channels=1,
        #    kernel_size=kernel_size
        #)
        #self.conv5 = nn.Conv2d(
        #    in_channels=1,
        #    out_channels=1,
        #    kernel_size=kernel_size
        #)
        # Dropout for convolutions
        self.drop = nn.Dropout2d()

        self.features = {}

        return None

    def init_optimizer(self, lr, momentum):
        self.optimizer = torch.optim.SGD(
                self.parameters(), 
                lr=lr, 
                momentum=momentum
            )
        return None

    def make_fc1(self, x):
        if not hasattr(self, 'fc1'):
            length = x.shape[1]
            self.fc1 = nn.Linear(length, 50)
        return None

    def make_fc2(self, x):
        if not hasattr(self, 'fc2'):
            length = x.shape[1]
            self.fc2 = nn.Linear(length, 300)
        return None

    def make_fc3(self, x):
        if not hasattr(self, 'fc3'):
            length = x.shape[1]
            self.fc3 = nn.Linear(length, self.N_out_channels)
        return None

    def forward(self, x):
        x = self.conv1(x) # 1
        #x = self.drop(x) # 5
        #x = torch.nn.functional.max_pool2d(x, kernel_size=2) # 2
        x = self.activation(x)

        x = self.conv2(x) # 4
        #x = self.drop(x) # 5
        #x = torch.nn.functional.max_pool2d(x, kernel_size=2) # 6
        x = self.activation(x)

        #x = self.conv3(x)
        #x = self.drop(x) # 5
        #x = torch.nn.functional.max_pool2d(x, kernel_size=2) # 2
        #x = self.activation(x)

        #x = self.conv4(x)
        #x = self.drop(x) # 5
        #x = torch.nn.functional.max_pool2d(x, kernel_size=2) # 2
        #x = self.activation(x)

        #x = self.conv5(x)
        #x = self.drop(x) # 5
        #x = torch.nn.functional.max_pool2d(x, kernel_size=2) # 2
        #x = self.activation(x)

        x = x.flatten(start_dim=1) # 8
        
        #self.make_fc1(x)
        #x = self.fc1(x) # 9
        #x = self.activation(x)

        #plt.hist(x.flatten().detach().cpu().numpy())
        #plt.show()

        #self.make_fc2(x)
        #x = self.fc2(x) # 11
        #x = self.activation(x)

        self.make_fc3(x)
        x = self.fc3(x) # 11
        x = self.activation(x)
        
        return x

    def get_features(self, name):
        def hook(module, input, output):
            self.features[name] = output.detach()
        return hook

    def register_feature_hooks(self):
        self.conv1.register_forward_hook(self.get_features('conv1'))
        return None

    def save(self, epoch, train_loss, test_loss):
        import os
        import time
        import math

        start = time.time()

        if os.path.isfile('./sate.tar'):
            checkpoints = torch.load('./state.tar', weights_only=True)
        else:
            checkpoints = {}

        checkpoints[epoch] = {
            'train_loss': train_loss,
            'test_loss': test_loss,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoints, './state.tar')

        end = time.time()
        elapsed = end - start
        minutes = math.floor(elapsed / 60.)
        print('{0:0.0f} min, {1:0.1f} s to save epoch {2:0.0f}'.format(
            minutes, 
            elapsed - minutes * 60.,
            epoch
        ))

        return None

    def load(self):
        checkpoints = torch.load('./state.tar', weights_only=True)
        epochs = np.array(list(checkpoints.keys()))
        last_epoch = epochs.max()
        self.load_state_dict(checkpoints[last_epoch]['model_state_dict'])
        self.optimizer.load_state_dict(
            checkpoints[last_epoch]['optimizer_state_dict']
        )
        return None

def main(Nfiles=None):
    import preprocessing
    import random
    import wandb

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    import torchvision
    import torch.nn as nn

    try:
        has_mps = torch.backends.mps.is_available()
    except:
        has_mps = False
    if has_mps:
        device_str = 'mps'
    elif torch.cuda.is_available():
        device_str = 'cuda'
    else:
        device_str = 'cpu'
    device = torch.device(device_str)
    torch.set_default_device(device_str)

    # Things wandb will track
    lr=0.01 # learning rate
    momentum = 0.5
    N_batches = 20
    N_epochs = 4
    kernel_size = 80 
    activation =  torch.nn.functional.relu
    N_conv1_out_chan = 50
    N_conv2_out_chan = 1

    # Other things
    N_out_channels = 1

    wandb.init(
        # set the wandb project where this run will be logged
        project="2d_gallearn",
        name='test',

        # track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            'momentum': momentum,
            'activation_func': activation,
            "architecture": "CNN",
            "dataset": (
                "500x500 hosts, xy projection, drop >2000x2000"
            ),
            "epochs": N_epochs,
            'batches': N_batches,
            'kernel size': kernel_size,
            'N_conv_layers': 2,
            'N_fc_layers': 1,
            'N_conv1_out_channels': N_conv1_out_chan,
            'N_conv2_out_channels': N_conv2_out_chan,
        }
    )

    d = preprocessing.load_data('gallearn_data_500x500_2d_tgt.h5')
    X = d['X'].to(device=device_str)
    X = preprocessing.new_min_max_scale(X)[:Nfiles]
    ys = d['ys_sorted'].to(device=device_str)[:Nfiles]

    N_all = len(ys) 
    print('{0:0.0f} galaxies in data'.format(N_all))
    N_test = max(1, int(0.15 * N_all))
    N_train = N_all - N_test

    indices_test = np.random.randint(0, N_all, N_test)
    is_train = np.ones(N_all, dtype=bool)
    is_train[indices_test] = False
    ys_train = ys[is_train]
    ys_test = ys[~is_train]
    X_train = X[is_train]
    X_test = X[~is_train]

    # Run this once to load the train and test data straight into a dataloader 
    # class
    # that will provide the batches
    batch_size_train = max(1, int(N_train / N_batches))
    #batch_size_test = N_test
    batch_size_test = min(20, N_test)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, ys_train),
        batch_size=batch_size_train, 
        shuffle=True,
        generator=torch.Generator(device=device_str)
    )

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test, ys_test),
        batch_size=batch_size_test, 
        shuffle=True,
        generator=torch.Generator(device=device_str)
    )

    # Define the network.  This is a more typical way to define a network than 
    # the sequential structure.  We define a class for the network, and define 
    # the parameters in the constructor.  Then we use a function called forward 
    # to actually run the network.  It's easy to see how you might use residual 
    # connections in this format.

    # 1. A valid convolution with kernel size 5, 1 input channel and 10 output 
    #    channels
    # 2. A max pooling operation over a 2x2 area
    # 3. A Relu
    # 4. A valid convolution with kernel size 5, 10 input channels and 20 output 
    #    channels
    # 5. A 2D Dropout layer
    # 6. A max pooling operation over a 2x2 area
    # 7. A relu
    # 8. A flattening operation
    # 9. A fully connected layer mapping from (whatever dimensions we are at
    #    -- find 
    #    out using .shape) to 50
    # 10. A ReLU
    # 11. A fully connected layer mapping from 50 to 10 dimensions
    # 12. A softmax function.

    # He initialization of weights
    def weights_init(layer_in):
        if isinstance(layer_in, nn.Linear):
            nn.init.kaiming_uniform_(
                layer_in.weight,
                generator=torch.Generator(device=device_str)
            )
            layer_in.bias.data.fill_(0.0)
        return None

    # Create network
    model = Net(
            activation,
            kernel_size,
            N_conv1_out_chan,
            N_conv2_out_chan,
            N_out_channels 
        ).to(device)
    # Initialize model weights
    model.apply(weights_init)
    model.init_optimizer(lr, momentum)

    loss_function = torch.nn.MSELoss()

    # Main training routine
    def train(epoch):
        model.train()
        # Get each
        sum_losses = 0.
        N_optimized = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            model.optimizer.zero_grad()
            output = model(data.to(device))
            loss = loss_function(output, target)
            loss.backward()
            model.optimizer.step()
            # Store results
            sum_losses += loss
            N_optimized += len(data)

            #if batch_idx % 5 == 0:
            if True:
                print(
                    '\nTrain Epoch: {0}'
                            '[{1}/{2} samples optimized]'
                            '\tLoss: {3:.6f}'.format(
                        epoch, 
                        N_optimized,
                        len(train_loader.dataset), 
                        loss.item()
                    )
                )
                feedback = torch.stack(
                    (output.flatten(), target.flatten()),
                    dim=1
                )
                df = pd.DataFrame(
                    data=feedback.detach().cpu().numpy(),
                    columns=['Predictions', 'Targets']
                )
                pd.options.display.float_format = '{:,.3f}'.format
                print('Training batch results:')
                print(df)
        avg_loss = sum_losses / (batch_idx + 1)
        wandb.log({'training loss': avg_loss})
        return avg_loss 

    # Run on test data
    def test():
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                output = model(data.to(device))
                batch_loss = loss_function(output, target).item()
                test_loss += batch_loss

                if i == 0:
                    feedback = torch.stack(
                        (output.flatten(), target.flatten()),
                        dim=1
                    )
                    df = pd.DataFrame(
                        data=feedback.detach().cpu().numpy(),
                        columns=['Predictions', 'Targets']
                    )
                    pd.options.display.float_format = '{:,.3f}'.format
                    print('\nA batch of test results:')
                    print(df)
        test_loss /= i + 1
        print('\nTest set: Avg. loss: {:.4f}\n'.format(
            test_loss
        ))
        return test_loss 

    # Get initial performance
    print('\nGetting initial performance.')
    test()
    print('Training.')
    for epoch in range(1, N_epochs + 1):
        train_loss = train(epoch)
        test_loss = test()
        model.save(epoch, train_loss, test_loss)

    # Run network on data we got before and show predictions
    test_examples = enumerate(test_loader)
    batch_idx, (test_data, test_tgts) = next(test_examples)
    output = model(test_data.to(device))
    print('\nSome test outputs:')
    print(output)
    print('Corresponding targets:')
    print(test_tgts)

    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #y_pred = model(X)
    #ax.hist(model[0].input.reshape(-1), bins=25, histtype='step')
    #ax.set_yscale('log')
    #plt.show()

    return model 

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run the galaxy shape CNN.')
    parser.add_argument(
        '-n',
        '--num-gals',
        type=int,
        required=False, 
        help=(
            'The number of galaxies to use in the network. If not specified,'
            ' the network will use all galaxies available.'
        )
    )
    args = parser.parse_args()
    Nfiles = args.num_gals
    main(Nfiles)
