import torch.nn as nn
import torch
import datetime

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

def save_wandb_id(wandb):
    import paths
    import os
    with open(os.path.join(paths.data, wandb.run.name + '_id.txt'), 'w') as f:
        f.write(wandb.run.id)
    return None

def load_wandb_id(run_name):
    import paths
    import os
    with open(os.path.join(paths.data, run_name + '_id.txt'), 'r') as f:
        wandb_id = f.read()
    return wandb_id

def load_net(run_name):
    import pickle
    import paths
    import os
    with open(os.path.join(paths.data, run_name + '_args' + '.pkl'), 
              'rb') as f:
        args_dict = pickle.load(f)
    print(args_dict)
    args = []
    for key in ['activation_module',
                'kernel_size',
                'conv_channels',
                'N_groups',
                'p_fc_dropout',
                'N_out_channels',
                'lr',
                'momentum']:
        if key not in args_dict:
            args_dict[key] = None
    args_dict['run_name'] = run_name
    model = Net(**args_dict)
    model.init_optimizer()
    model.load()
    return model

def find_closest_N_groups(N_channels, N_groups):
    # Get all divisors of N_channels
    divisors = [i for i in range(1, N_channels + 1) if N_channels % i == 0]
    # Find the divisor closest to N_groups
    closest_divisor = min(divisors, key=lambda x: abs(x - N_groups))
    return closest_divisor

class Net(nn.Module):
    def __init__(
                self,
                activation_module,
                kernel_size,
                conv_channels,
                N_groups,
                p_fc_dropout,
                N_out_channels,
                lr,
                momentum,
                run_name
            ):
        import paths
        import os

        super(Net, self).__init__()

        self.state_path = os.path.join(
            paths.data, 
            run_name + '_state.tar'
        )

        self.last_epoch = 0

        self.activation_module = activation_module
        self.kernel_size = kernel_size
        self.conv_channels = conv_channels
        self.N_groups = N_groups
        self.p_fc_dropout = p_fc_dropout
        self.N_out_channels = N_out_channels
        self.momentum = momentum
        self.lr = lr
        self.run_name = run_name
        self.features = {}

        #----------------------------------------------------------------------
        # Define architecture
        #----------------------------------------------------------------------
        self.conv_block = nn.Sequential()
        in_channels = 1 
        i = 0
        for out_channels in conv_channels:
            self.conv_block.add_module(
                str(i),
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size
                )
            )
            i += 1
            if self.N_groups is not None:
                closest_N_groups = find_closest_N_groups(
                    out_channels, 
                    self.N_groups
                )
                self.conv_block.add_module(
                    str(i),
                    nn.GroupNorm(closest_N_groups, out_channels)
                )
                i += 1
            self.conv_block.add_module(
                str(i), 
                activation_module()
            )
            in_channels = out_channels
            i += 1
        if p_fc_dropout is not None and p_fc_dropout  0.:
            self.dropout = nn.Dropout1d(p_fc_dropout)
        self.fc_block = nn.Sequential(
            nn.LazyLinear(N_out_channels),
            activation_module()
        )

        # Dropout for convolutions
        #self.drop = nn.Dropout2d()

        return None

    def init_optimizer(self):
        self.optimizer = torch.optim.SGD(
                self.parameters(), 
                lr=self.lr, 
                momentum=self.momentum
            )
        return None

    def forward(self, x):
        x = self.conv_block(x)
        #x = self.drop(x) # 5
        #x = torch.nn.functional.max_pool2d(x, kernel_size=2) # 6

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
        x = self.dropout(x)
        x = self.fc_block(x)
        
        return x

    def register_feature_hooks(self):
        features = {}  # Dictionary to store outputs

        def hook_wrapper(output_dict, key):
            # Define the hook function
            def hook(module, input, output):
                # Traverse the keys to update the appropriate nested dictionary
                output_dict[key] = output
            return hook

        def process_module(module, output_dict):
            for name, layer in module._modules.items():
                if isinstance(layer, nn.Sequential) or len(layer._modules) > 0:
                    # Create a sub-dictionary for nested layers
                    output_dict[name] = {}
                    process_module(layer, output_dict[name]) # Recursion
                else:
                    # Register a hook for non-nested layers
                    layer_type = str(layer).split('(')[0]
                    key = ':'.join([name, layer_type])
                    layer.register_forward_hook(hook_wrapper(output_dict, key))

        process_module(self, features)
        self.features = features
        return None

    def save_args(self):
        import pickle
        import paths
        import os
        args = {
            'activation_module': self.activation_module,
            'kernel_size': self.kernel_size,
            'conv_channels': self.conv_channels,
            'N_groups': self.N_groups,
            'p_fc_dropout': self.p_fc_dropout,
            'N_out_channels': self.N_out_channels,
            'lr': self.lr,
            'momentum': self.momentum
        }
        with open(os.path.join(paths.data, self.run_name + '_args' + '.pkl'), 
                  'wb') as f:
            pickle.dump(args, f, protocol=pickle.HIGHEST_PROTOCOL)
        return None

    def save_state(self, epoch, train_loss, test_loss):
        import os
        import time
        import math

        start = time.time()

        if os.path.isfile(self.state_path):
            checkpoints = torch.load(self.state_path, weights_only=True)
        else:
            checkpoints = {}

        checkpoints[epoch] = {
            'train_loss': train_loss,
            'test_loss': test_loss,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoints, self.state_path)

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
        import numpy as np
        checkpoints = torch.load(self.state_path, weights_only=True)
        epochs = np.array(list(checkpoints.keys()))
        self.last_epoch = epochs.max()
        self.load_state_dict(checkpoints[self.last_epoch]['model_state_dict'])
        self.optimizer.load_state_dict(
            checkpoints[self.last_epoch]['optimizer_state_dict']
        )
        return None

def main(Nfiles=None, wandb_mode='n', run_name=None):
    if wandb_mode == 'r' and run_name is None:
        raise Exception(
            'User must provide a `run_name` if `wandb_mode` is \'r\' for'
            ' resume.'
        )
    import preprocessing
    import random
    import wandb
    import os
    import paths

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

    def weights_init(layer_in):
        '''
        Kaiming He initialization of weights
        '''
        if isinstance(layer_in, nn.Linear):
            nn.init.kaiming_uniform_(
                layer_in.weight,
                generator=torch.Generator(device=device_str)
            )
            layer_in.bias.data.fill_(0.0)
        return None

    def train(epoch):
        model.train()
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
        return avg_loss 

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

    ###########################################################################
    # Load the data
    ###########################################################################

    # Hardcoding the dataset like this instead of getting it from a model
    # attribute could potentially cause problems if the model the code
    # loads was supposed to use a different dataset. We'll deal with that if it
    # ever happens.
    dataset = 'gallearn_data_256x256_2d_tgt.h5'

    d = preprocessing.load_data(dataset)
    X = d['X'].to(device=device_str)
    # Linearly min-max scale the data from 0 to 255.
    X = preprocessing.new_min_max_scale(X)[:Nfiles]
    ys = d['ys_sorted'].to(device=device_str)[:Nfiles]

    N_all = len(ys) 
    print('{0:0.0f} galaxies in data'.format(N_all))
    N_test = max(1, int(0.15 * N_all))
    N_train = N_all - N_test

    # Train-test split
    indices_test = np.random.randint(0, N_all, N_test)
    is_train = np.ones(N_all, dtype=bool)
    is_train[indices_test] = False
    ys_train = ys[is_train]
    ys_test = ys[~is_train]
    X_train = X[is_train]
    X_test = X[~is_train]

    ###########################################################################

    N_epochs = 50 
    N_batches = 20
    loss_function = torch.nn.MSELoss()

    ###########################################################################
    # Build (or rebuild) the model
    ###########################################################################
    if wandb_mode == 'n' and run_name is None:
        run_name = datetime.datetime.today().strftime('%Y%m%d')
    if run_name is not None and os.path.isfile(
                os.path.join(paths.data, run_name + '_state.tar')
            ):
        model = load_net(run_name)
        if wandb_mode == 'r':
            run_id = load_wandb_id(run_name)
            wandb.init(
                project='2d_gallearn',
                id=run_id,
                resume='must'
            )
    elif wandb_mode == 'r':
        raise Exception(
            '`wandb_mode` is set to resume, but there is no corresponding'
            ' state file'
        )
    else:
        # If the state file doesn't exist

        # Things wandb will track
        lr = 0.00001 # learning rate
        momentum = 0.5
        kernel_size = 40
        activation_module = nn.ReLU
        conv_channels = [50, 25, 10, 3, 1]
        N_groups = 4
        p_fc_dropout = 0.5

        # Other things
        N_out_channels = 1

        if wandb_mode == 'y':
            wandb.init(
                # Set the wandb project where this run will be logged.
                project="2d_gallearn",

                # Track hyperparameters and run metadata.
                config={
                    "learning_rate": lr,
                    'momentum': momentum,
                    'activation_func': activation_module,
                    "dataset": dataset,
                    'batches': N_batches,
                    'kernel size': kernel_size,
                    'N_fc_layers': 1,
                    'conv_channels': conv_channels,
                    'N_groups': N_groups,
                    'p_fc_dropout': p_fc_dropout
                }
            )
            run_name = wandb.run.name
            save_wandb_id(wandb)

        # Define the model if the we didn't rebuild one from a argument and
        # state files.
        model = Net(
                activation_module,
                kernel_size,
                conv_channels,
                N_groups,
                p_fc_dropout,
                N_out_channels,
                lr,
                momentum,
                run_name
            ).to(device)
        model.save_args()
        model(X[:1]) # Run a dummy fwd pass to initialize any lazy layers.
        model.init_optimizer()
        model.apply(weights_init) # Init model weights.
    
        if wandb_mode == 'y':
            wandb.config['architecture'] = repr(model)

    print(model)
    
    ###########################################################################
    # Make the DataLoaders.
    ###########################################################################
    batch_size_train = max(1, int(N_train / N_batches))
    batch_size_test = min(N_batches, N_test)
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
    ###########################################################################

    print('\nGetting initial performance.')
    test()

    print('Training.')
    for epoch in range(model.last_epoch + 1, model.last_epoch + N_epochs + 1):
        train_loss = train(epoch)
        test_loss = test()
        if wandb_mode in ['y', 'r']:
            wandb.log({'training loss': train_loss,
                       'test loss': test_loss})
        model.save_state(epoch, train_loss, test_loss)

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
    parser.add_argument(
        '-w',
        '--wandb',
        type=str,
        choices=['n', 'y', 'r'],
        default='n',
        help=(
            '`wandb` mode. Choices are'
            ' \'n\': No'
            ' interaction.'
            ' \'y\': Yes, start a new run.'
            ' \'r\': Resume a run.'
        )
    )
    parser.add_argument(
        '-r',
        '--run-name',
        type=str,
        help=(
            'The name to give a new run, or the name of the run to resume.'
            ' (Required'
            ' if --wandb is \'r\')'
        )
    )

    args = parser.parse_args()
    if args.wandb == 'r' and args.run_name is None:
        parser.error('--run-name is required when --wandb is \'r\'')

    Nfiles = args.num_gals
    wandb_mode = args.wandb
    run_name = args.run_name
    main(Nfiles, wandb_mode=wandb_mode, run_name=run_name)
