def main(Nfiles=None):
    #from julia.api import Julia
    #jl = Julia(compiled_modules=False, debug=False)

    #import julia
    #from julia import Pkg
    #from julia import Main

    import preprocessing
    import random
    import wandb

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    import torch
    import torchvision
    import torch.nn as nn
    import torch.optim as optim

    lr=0.01 # learning rate
    N_epochs = 4
    kernel_size = 20

    wandb.init(
        # set the wandb project where this run will be logged
        project="gallearn",

        # track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "architecture": "CNN",
            "dataset": "500x500 hosts, xy projection, drop >2000x2000",
            "epochs": N_epochs,
            'kernel size': kernel_size
        }
    )


    #Pkg.activate('/export/nfs0home/pstaudt/projects/gal-learn/GalLearn')

    # Attempt to locate the Julia executable path
    #import subprocess
    #try:
    #    julia_path = subprocess.check_output(
    #            ["which", "julia"]
    #        ).decode("utf-8").strip()
    #    print("Julia executable path:", julia_path)
    #    
    #    # Optional: Print Julia version to verify
    #    julia_version = subprocess.check_output(
    #            [julia_path, "--version"]
    #        ).decode("utf-8").strip()
    #    print("Julia version:", julia_version)
    #except subprocess.CalledProcessError:
    #    print("Julia executable not found in PATH")

    #Main.include('image_loader.jl')
    #obs_sorted, ys, ys_sorted, X, files = julia.Main.image_loader.load_data(
    #    Nfiles=Nfiles
    #)

    d = preprocessing.load_data()
    X = d['X']
    X = preprocessing.min_max_scale(X)
    ys = d['ys_sorted']

    N_all = len(ys) 
    print('{0:0.0f} galaxies in data'.format(N_all))
    N_test = max(1, int(0.15 * N_all))
    N_train = N_all - N_test
    N_batches = 20

    indices_test = np.random.randint(0, N_all, N_test)
    is_train = np.ones(N_all, dtype=bool)
    is_train[indices_test] = False
    ys_train = ys[is_train]
    ys_test = ys[~is_train]
    X_train = X[is_train]
    X_test = X[~is_train]

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

    # Run this once to load the train and test data straight into a dataloader 
    # class
    # that will provide the batches
    batch_size_train = max(1, int(N_train / N_batches))
    batch_size_test = N_test
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

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(
                in_channels=3,
                out_channels=10,
                kernel_size=kernel_size
            )
            self.conv2 = nn.Conv2d(
                in_channels=10,
                out_channels=20,
                kernel_size=kernel_size
            )
            self.conv3 = nn.Conv2d(
                in_channels=20,
                out_channels=50,
                kernel_size=kernel_size
            )
            self.conv4 = nn.Conv2d(
                in_channels=50,
                out_channels=80,
                kernel_size=kernel_size
            )
            self.conv5 = nn.Conv2d(
                in_channel=80,
                out_channel=80,
                kernel_size=kernel_dize
            )
            # Dropout for convolutions
            self.drop = nn.Dropout2d()
            # Fully connected layer
            self.fc2 = nn.Linear(50, 2)
            return None

        def make_fc1(self, x):
            length = x.shape[1]
            self.fc1 = nn.Linear(length, 50)
            return None

        def forward(self, x):
            # print('x type: {}'.format(type(x)))
            # print('x dtype: {}'.format(x.dtype))
            # print('x device: {}'.format(x.device))
            x = self.conv1(x) # 1
            x = torch.nn.functional.max_pool2d(x, kernel_size=2) # 2
            x = torch.nn.functional.relu(x) # 3

            x = self.conv2(x) # 4
            x = self.drop(x) # 5
            x = torch.nn.functional.max_pool2d(x, kernel_size=2) # 6
            x = torch.nn.functional.relu(x) # 7

            x = self.conv3(x)
            x = torch.nn.functional.relu(x)

            x = self.conv4(x)
            x = torch.nn.functional.relu(x)

            x = self.conv5(x)
            x = torch.nn(functional.relu(x))

            x = x.flatten(start_dim=1) # 8
            self.make_fc1(x)
            x = self.fc1(x) # 9
            x = torch.nn.functional.relu(x) # 10
            x = self.fc2(x) # 11
            x = torch.nn.functional.sigmoid(x) # 12
            
            return x

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
    model = Net().to(device)
    # Initialize model weights
    model.apply(weights_init)
    # Define optimizer
    optimizer = optim.SGD(
        model.parameters(), 
        lr=lr, 
        momentum=0.5
    )

    loss_function = torch.nn.MSELoss()

    # Main training routine
    def train(epoch):
        model.train()
        # Get each
        sum_losses = 0.
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data.to(device))
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            # Store results
            sum_losses += loss
            if batch_idx % 5 == 0:
                print(
                    'Train Epoch: {0}'
                            '[{1}/{2} samples optimized]'
                            '\tLoss: {3:.6f}'.format(
                        epoch, 
                        batch_idx * len(data), 
                        len(train_loader.dataset), 
                        loss.item()
                    )
                )
        avg_loss = sum_losses / (batch_idx + 1)
        wandb.log({'training loss': avg_loss})
        return None

    # Run on test data
    def test():
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data.to(device))
                test_loss += loss_function(output, target).item()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Avg. loss: {:.4f}\n'.format(
            test_loss
        ))
        print('10 SAMPLE OUTPUTS')
        print(output[:10])
        print('\nCORRESPONDING TARGETS')
        print(target[:10])
        return None

    # Get initial performance
    print('\nGetting initial performance.')
    test()
    print('Training.')
    for epoch in range(1, N_epochs + 1):
        train(epoch)
        test()

    # Run network on data we got before and show predictions
    test_examples = enumerate(test_loader)
    batch_idx, (test_data, test_tgts) = next(test_examples)
    output = model(test_data.to(device))
    print(output)
    print(test_tgts)
    
    return None

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
