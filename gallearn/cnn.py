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


def get_radii(d):
    import h5py
    import os
    import numpy as np
    import pandas as pd

    from . import config

    dtype_dict = {'galaxyID': int}
    df_Re_host = pd.read_csv(
        config.config[f'{__package__}_paths']['host_2d_shapes'],
        #dtype=dtype_dict
    )
    assert not df_Re_host.duplicated(
        subset=['galaxyID', 'view', 'band']
    ).any(), (
        'The host shapes file has duplicate rows..'
    )
    df_Re_sat = pd.read_csv(
        config.config[f'{__package__}_paths']['sat_2d_shapes'],
        dtype=dtype_dict
    )
    assert not df_Re_sat.duplicated(
        subset=['galaxyID', 'view', 'band']
    ).any(), 'The satellite shapes file has duplicate rows.'
    df_Re = pd.concat([df_Re_host, df_Re_sat], axis=0)
    df_Re.rename(
        columns={'galaxyID': 'id', 'view': 'orientation'},
        inplace=True
    )
    # Need to isolate the r-band, otherwise there will be id-oreintation
    # duplicates.
    df_Re = df_Re.loc[df_Re['band'] == 'band_r']
    assert not df_Re.duplicated(
        subset=['id', 'orientation']
    ).any(), (
        'The combined host + satellite shapes DataFrame has duplicate rows.'
    )

    ids_X = np.char.replace(
        d['obs_sorted'].astype(str),
        'object_',
        ''
    ).astype(int)
    df_X = pd.DataFrame({'id': ids_X, 'orientation': d['orientations']})

    df = df_X.merge(
        df_Re,
        left_on=['id', 'orientation'],
        right_on=['id', 'orientation'],
        how='left',
        # There should not be any duplicate rows in either df_X or df_Re.
        validate='one_to_one'
    )

    # Ensure that the DataFrame from which we'll grab the radii is lined up
    # exactly with the inputted dictionary.
    pd.testing.assert_frame_equal(df_X, df[['id', 'orientation']])

    rs = torch.tensor(
        df['Re'].values,
        dtype=torch.float32
    ).unsqueeze(1)

    return rs 


def save_wandb_id(wandb):
    from . import config
    import os
    with open(
            os.path.join(
                config.config[f'{__package__}_paths']['project_data_dir'],
                wandb.run.name,
                'id.txt'),
            'w') as f:
        f.write(wandb.run.id)
    return None


def load_wandb_id(run_name):
    from . import config
    import os
    with open(
            os.path.join(
                config.config[f'{__package__}_paths']['project_data_dir'],
                run_name,
                'id.txt'),
            'r') as f:
        wandb_id = f.read()
    return wandb_id


def find_closest_N_groups(N_channels, N_groups):
    # Get all divisors of N_channels
    divisors = [i for i in range(1, N_channels + 1) if N_channels % i == 0]
    # Find the divisor closest to N_groups
    closest_divisor = min(divisors, key=lambda x: abs(x - N_groups))
    return closest_divisor


class Net(nn.Module):
    def __init__(
                self,
                kernel_size,
                conv_channels,
                N_groups,
                N_out_channels,
                lr,
                momentum,
                run_name,
                dataset,
                scaling_function
            ):
        from . import config
        import os
        import numpy as np

        super(Net, self).__init__()

        self.state_path = os.path.join(
            config.config[f'{__package__}_paths']['project_data_dir'],
            run_name + '_state.tar'
        )

        self.last_epoch = 0

        self.activation_module = nn.ReLU
        self.kernel_size = kernel_size
        self.conv_channels = conv_channels
        self.N_groups = N_groups
        self.N_out_channels = N_out_channels
        self.momentum = momentum
        self.lr = lr
        self.run_name = run_name
        self.dataset = dataset
        self.scaling_function = scaling_function
        self.features = {}

        self.run_dir = os.path.join(config.config[f'{__package__}_paths']['project_data_dir'], self.run_name)
        self.states_dir = os.path.join(self.run_dir, 'states')
        if not os.path.isdir(self.run_dir):
            os.mkdir(self.run_dir)
            os.mkdir(self.states_dir)

        #----------------------------------------------------------------------
        # Define architecture
        #----------------------------------------------------------------------
        self.backbone = nn.Sequential()
        in_channels = 3
        i = 0
        for out_channels in conv_channels:
            self.backbone.add_module(
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
                self.backbone.add_module(
                    str(i),
                    nn.GroupNorm(closest_N_groups, out_channels)
                )
                i += 1
            self.backbone.add_module(
                str(i), 
                self.activation_module()
            )
            in_channels = out_channels
            i += 1

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Sequential()
        module_i = 0
        for layer_i in range(3):
            # Make all but the last layer of the 4-layer head
            power = np.log2(in_channels) - 1.
            out_channels = int(2. ** power)
            # Note that, due to late fusion, `in_channels` may be 1 less than 
            # the actual channels
            # going into the first layer of the head, but it doesn't matter; 
            # we're
            # just using `in_channels` to step down by one power of 2.
            self.head.add_module(
                str(module_i), 
                nn.LazyLinear(out_channels)
            )
            module_i += 1
            self.head.add_module(
                str(module_i),
                nn.BatchNorm1d(out_channels)
            )
            module_i += 1
            self.head.add_module(
                str(module_i),
                self.activation_module()
            )
            module_i += 1
            in_channels = out_channels

        self.head.add_module(
            str(module_i),
            nn.Linear(in_channels, self.N_out_channels)
        )

        return None

    def init_optimizer(self):
        self.optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=self.lr, 
            )
        return None

    def forward(self, x, rs):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.flatten(start_dim=1) # 8
        x = nn.functional.dropout(x, 0.2)
        x = torch.cat((x, rs), dim=1)
        x = self.head(x)
        
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
        from . import config
        import os
        args = {
            'activation_module': self.activation_module,
            'kernel_size': self.kernel_size,
            'conv_channels': self.conv_channels,
            'N_groups': self.N_groups,
            'N_out_channels': self.N_out_channels,
            'lr': self.lr,
            'momentum': self.momentum,
            'net_type': 'original',
            'dataset': self.dataset,
            'scaling_function': self.scaling_function
        }
        with open(os.path.join(config.config[f'{__package__}_paths']['project_data_dir'], self.run_name + '_args' + '.pkl'), 
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


def _replace_first_conv_in_channels(module, in_channels):
    """
    Recursively find the first Conv2d layer in a module and replace it
    with a new Conv2d that has the specified in_channels.
    Returns True if a replacement was made, False otherwise.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
                padding_mode=child.padding_mode,
            )
            setattr(module, name, new_conv)
            return True
        if _replace_first_conv_in_channels(child, in_channels):
            return True
    return False


def _make_backbone_feature_extractor(module):
    """
    Modify common backbone architectures (ResNet, VGG, etc.) to output
    feature maps instead of class predictions by removing the final
    pooling and fully connected layers.
    """
    # ResNet-style: has 'fc' and 'avgpool' attributes
    if hasattr(module, 'fc') and isinstance(module.fc, nn.Linear):
        module.fc = nn.Identity()
        # Also need to override forward to skip the flatten call
        if hasattr(module, 'avgpool'):
            module.avgpool = nn.Identity()

            # Store original forward and create new one that skips flatten
            original_forward = module.forward

            def new_forward(x):
                x = module.conv1(x)
                x = module.bn1(x)
                x = module.relu(x)
                x = module.maxpool(x)
                x = module.layer1(x)
                x = module.layer2(x)
                x = module.layer3(x)
                x = module.layer4(x)
                return x

            module.forward = new_forward
        return True
    # VGG-style: has 'classifier' attribute
    if hasattr(module, 'classifier'):
        module.classifier = nn.Identity()
        if hasattr(module, 'avgpool'):
            module.avgpool = nn.Identity()
        return True
    return False


class BernoulliNet(nn.Module):
    def __init__(
                self,
                lr,
                momentum,
                backbone,
                dataset,
                in_channels=None,
            ):
        from . import preprocessing

        super(BernoulliNet, self).__init__()
        self.lr = lr
        self.momentum = momentum
        self.backbone = backbone
        self.dataset = dataset

        _make_backbone_feature_extractor(self.backbone)

        if in_channels is not None:
            if not _replace_first_conv_in_channels(self.backbone, in_channels):
                raise ValueError("No Conv2d layer found in backbone")

        self.scaling_function = preprocessing.sasinh_imgs_sscale_vmaps

        self.head = nn.Sequential(
            nn.LazyLinear(256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 1),
        )

        return None

    def init_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.lr, 
        )
        return None

    def forward(self, x, rs):
        x = self.backbone(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(start_dim=1)
        x = torch.cat((x, rs), dim=1)
        x = self.head(x)
        return x


class ResNet(nn.Module):
    def __init__(
                self,
                run_name,
                N_out_channels=None,
                lr=None,
                momentum=None,
                resblock=None,
                n_blocks_list=None,
                dataset=None,
                out_channels_list=[64, 128, 256, 512],
                N_img_channels=None
            ):
        '''
        Adapted from https://github.com/freshtechyy/resnet.git

        Parameters
        ----------
            resblock: residual block type, BasicResBlock for ResNet-18, 34 or 
                      BottleNeck for ResNet-50, 101, 152
            n_class: number of classes for image classifcation (used in
                classfication head)
            n_blocks_list: number of residual blocks for each conv layer 
                (conv2_x - conv5_x)
            out_channels_list: list of the output channel numbers for conv2_x 
                - conv5_x
            N_img_channels: the number of channels of input image
        '''
        from . import config
        import os
        from . import preprocessing

        super(ResNet, self).__init__()

        self.run_name = run_name

        self.run_dir = os.path.join(
            config.config[f'{__package__}_paths']['project_data_dir'],
            run_name
        )
        self.states_dir = os.path.join(self.run_dir, 'states')
        if not os.path.isdir(self.run_dir):
            os.mkdir(self.run_dir)
            os.mkdir(self.states_dir)
            self.need_to_load = False
        else:
            self.need_to_load = True

        if self.need_to_load and (
                    N_out_channels is not None
                    or lr is not None
                    or momentum is not None
                    or resblock is not None
                    or n_blocks_list is not None
                    or dataset is not None
                    or N_img_channels is not None
                ):
            raise Exception(
                'Run already exists but the user specified one or more'
                ' initialization arguments.'
            )

        if self.need_to_load:
            self.load_args()
        else:
            self.resblock = resblock
            self.N_out_channels = N_out_channels
            self.momentum = momentum
            self.lr = lr
            self.n_blocks_list = n_blocks_list
            self.out_channels_list = out_channels_list
            self.N_img_channels = N_img_channels
            self.dataset = dataset

            self.last_epoch = 0

        self.scaling_function = preprocessing.sasinh_imgs_sscale_vmaps
        self.activation_module = nn.ReLU

        self.features = {}

        #----------------------------------------------------------------------
        # Define architecture
        #----------------------------------------------------------------------
        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=N_img_channels, 
                out_channels=out_channels_list[0],
                kernel_size=7,
                stride=2,
                padding=3
            ),
            nn.BatchNorm2d(out_channels_list[0]),
            self.activation_module(),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1
            )
        )

        # Create four convoluiontal layers
        in_channels = out_channels_list[0] 
        # For the first block of the second layer, do not downsample and use 
        # stride=1.
        self.conv2_x = self.CreateLayer(
            self.resblock,
            self.n_blocks_list[0], 
            in_channels,
            out_channels_list[0],
            stride=1
        )
        
        # For the first blocks of conv3_x - conv5_x layers, perform 
        # downsampling using stride=2.
        # By default, resblock.expansion = 4 for ResNet-50, 101, 152, 
        # resblock.expansion = 1 for ResNet-18, 34.
        self.conv3_x = self.CreateLayer(
            self.resblock, self.n_blocks_list[1], 
            out_channels_list[0]*self.resblock.expansion,
            out_channels_list[1],
            stride=2
        )
        self.conv4_x = self.CreateLayer(
            self.resblock,
            self.n_blocks_list[2],
            out_channels_list[1]*self.resblock.expansion,
            out_channels_list[2],
            stride=2
        )
        self.conv5_x = self.CreateLayer(
            self.resblock,
            self.n_blocks_list[3], 
            out_channels_list[2]*self.resblock.expansion,
            out_channels_list[3],
            stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Head
        self.head = nn.Sequential(
            nn.LazyLinear(2048),
            nn.BatchNorm1d(2048),
            self.activation_module(),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            self.activation_module(),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            self.activation_module(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            self.activation_module(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            self.activation_module(),

            nn.Linear(64, self.N_out_channels),
            #nn.Sigmoid()
            #nn.ReLU()
        )

        if self.need_to_load:
            self.init_optimizer()
            self.load()
            self.need_to_load = False

        return None

    def init_optimizer(self):
        self.optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=self.lr, 
            )
        return None

    def forward(self, x, rs):
        """
        Args: 
            x: input image
        Returns:
            x: target prediction
        """
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)

        # Head
        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = torch.nn.functional.dropout(x, 0.99)
        x = torch.cat((x, rs), dim=1)
        x = self.head(x)

        return x

    def CreateLayer(
                self,
                resblock,
                n_blocks,
                in_channels,
                out_channels,
                stride=1
            ):
        """
        Create a layer with specified type and number of residual blocks.
        Args: 
            resblock: residual block type, BasicResBlock for ResNet-18, 34 or 
                      BottleNeck for ResNet-50, 101, 152
            n_blocks: number of residual blocks
            in_channels: number of input channels
            out_channels: number of output channels
            stride: stride used in the first 3x3 convolution of the first 
                resdiual block
            of the layer and 1x1 convolution for skip connection in that block
        Returns: 
            Convolutional layer
        """
        layer = []
        for i in range(n_blocks):
            if i == 0:
                # Downsample the feature map using input stride for the first
                # block of the layer.
                layer.append(resblock(
                    in_channels,
                    out_channels, 
                    self.activation_module,
                    stride=stride,
                    is_first_block=True
                ))
            else:
                # Keep the feature map size same for the rest three blocks of 
                # the layer.
                # by setting stride=1 and is_first_block=False.
                # By default, resblock.expansion = 4 for ResNet-50, 101, 152, 
                # resblock.expansion = 1 for ResNet-18, 34.
                layer.append(
                    resblock(
                        out_channels*resblock.expansion,
                        out_channels,
                        self.activation_module
                    )
                )

        return nn.Sequential(*layer)

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
        from . import config
        import os
        args = {
            'activation_module': self.activation_module,
            'N_out_channels': self.N_out_channels,
            'lr': self.lr,
            'momentum': self.momentum,
            'n_blocks_list': self.n_blocks_list,
            'N_img_channels': self.N_img_channels,
            'net_type': 'ResNet',
            'resblock': self.resblock,
            'dataset': self.dataset,
        }
        with open(os.path.join(config.config[f'{__package__}_paths']['project_data_dir'], self.run_name, 'args.pkl'), 
                  'wb') as f:
            pickle.dump(args, f, protocol=pickle.HIGHEST_PROTOCOL)
        return None

    def load_args(self):
        import pickle
        import os
        with open(os.path.join(self.run_dir, 'args.pkl'), 
                  'rb') as f:
            args_dict = pickle.load(f)
        print(args_dict)
        if 'net_type' not in args_dict:
            net_type = 'original'
        else:
            net_type = args_dict['net_type']
            del args_dict['net_type']
        args = []
        for key in [
                    'dataset',
                    'N_out_channels',
                    'lr',
                    'momentum',
                    'n_blocks_list'
                ]:
            if key not in args_dict:
                args_dict[key] = None
        args_dict['run_name'] = run_name

        self.N_out_channels = args_dict['N_out_channels']
        self.momentum = args_dict['momentum']
        self.lr = args_dict['lr']
        self.n_blocks_list = args_dict['n_blocks_list']
        self.N_img_channels = args_dict['N_img_channels']
        self.dataset = args_dict['dataset']
        self.resblock = args_dict['resblock']

        return None

    def save_state(self, epoch, train_loss, test_loss):
        import os
        import time
        import math

        start = time.time()

        checkpoint = {
            'train_loss': train_loss,
            'test_loss': test_loss,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(
            checkpoint, 
            os.path.join(
                    self.states_dir,
                    'epoch{0:03}_state.tar'.format(epoch)
                )
        )

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
        import re
        import os

        states = os.listdir(self.states_dir)
        last_state_fname = max(states)
        # Get epoch number from file name by finding all numerals
        last_epoch = int(re.findall(r'\d+', last_state_fname)[0])
        self.last_epoch = last_epoch

        checkpoint = torch.load(
            os.path.join(self.states_dir, last_state_fname),
            weights_only=True
        )

        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(
            checkpoint['optimizer_state_dict']
        )
        return None


class BottleNeck(nn.Module):
    # Scale factor of the number of output channels
    expansion = 4

    def __init__(
                self,
                in_channels,
                out_channels, 
                activation_module,
                stride=1,
                is_first_block=False
            ):
        """
        Args: 
            in_channels: number of input channels
            out_channels: number of output channels
            stride: stride using in (a) 3x3 convolution and 
                    (b) 1x1 convolution used for downsampling for skip 
                    connection
            is_first_block: whether it is the first residual block of the layer
        """
        super().__init__()

        self.activation_module = activation_module
        self.activation_function = activation_module()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels*self.expansion,
                               kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)

        # Skip connection goes through 1x1 convolution with stride=2 for 
        # the first blocks of conv3_x, conv4_x, and conv5_x layers for matching
        # spatial dimension of feature maps and number of channels in order to 
        # perform the add operations.
        self.downsample = None
        if is_first_block:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels*self.expansion,
                        kernel_size=1,
                        stride=stride,
                        padding=0
                    ),
                nn.BatchNorm2d(out_channels*self.expansion)
            )
        return None

    def forward(self, x):
        """
        Args:
            x: input
        Returns:
            Residual block output
        """
        identity = x.clone()
        x = self.activation_function(self.bn1(self.conv1(x)))
        x = self.activation_function(self.bn2(self.conv2(x)))

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample:
            identity = self.downsample(identity)

        x += identity
        x = self.activation_function(x)

        return x


class BasicResBlock(nn.Module):
    # Scale factor of the number of output channels
    expansion = 1

    def __init__(
                self,
                in_channels,
                out_channels,
                activation_module,
                stride=1,
                is_first_block=False,
            ):
        """
        Adapted from https://github.com/freshtechyy/resnet.git

        Parameters
        ----------
            in_channels: number of input channels
            out_channels: number of output channels
            stride: stride using in (a) the first 3x3 convolution and 
                (b) 1x1 convolution used for downsampling for skip connection
            is_first_block: whether it is the first residual block of the layer
        """
        super().__init__()

        self.activation_module = activation_module
        self.activation_function = activation_module()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection goes through 1x1 convolution with stride=2 for 
        # the first blocks of conv3_x, conv4_x, and conv5_x layers for matching
        # spatial dimension of feature maps and number of channels in order to 
        # perform the add operations.
        self.downsample = None
        if is_first_block and stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                        in_channels=in_channels, 
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=stride,
                        padding=0
                    ),
                nn.BatchNorm2d(out_channels)
            )
        return None

    def forward(self, x):
        """
        Args:
            x: input
        Returns:
            Residual block ouput
        """
        identity = x.clone()
        x = self.activation_function(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.downsample:
            identity = self.downsample(identity)
        x += identity
        x = self.activation_function(x)

        return x


def main(Nfiles=None, wandb_mode='n', run_name=None):
    if wandb_mode == 'r' and run_name is None:
        raise Exception(
            'User must provide a `run_name` if `wandb_mode` is \'r\' for'
            ' resume.'
        )
    from . import preprocessing
    from . import config
    import random
    import wandb
    import os

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl

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
    # Note: Avoid torch.set_default_device() - it causes issues with MPS on macOS Tahoe

    def weights_init(layer_in):
        '''
        Kaiming He initialization of weights
        '''
        if isinstance(layer_in, nn.Linear):
            nn.init.kaiming_uniform_(
                layer_in.weight,
                generator=torch.Generator(device='cpu')
            )
            layer_in.bias.data.fill_(0.0)
        return None

    def train(epoch):
        model.train()
        sum_losses = 0.
        N_optimized = 0
        for batch_idx, (images, rs, target) in enumerate(train_loader):
            model.optimizer.zero_grad()
            output = model(images.to(device), rs.to(device))
            loss = loss_function(output, target)
            loss.backward()
            model.optimizer.step()

            # Store results
            sum_losses += loss
            N_optimized += len(images)

            #if batch_idx % 5 == 0:
            if True:
                print(
                    '\nTrain Epoch: {0}'
                            '[{1}/{2} samples optimized]'
                            '\tRMSE: {3:.6f}'.format(
                        epoch, 
                        N_optimized,
                        len(train_loader.dataset), 
                        np.sqrt(loss.item())
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
        with torch.no_grad():
            for i, (images, rs, target) in enumerate(test_loader):
                output = model(images.to(device), rs.to(device))
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
        print('\nTest set: RMSE: {:.4f}\n'.format(
            np.sqrt(test_loss)
        ))
        return test_loss 

    N_epochs = 200
    N_batches = 60 
    loss_function = torch.nn.MSELoss()

    ###########################################################################
    # Build (or rebuild) the model
    ###########################################################################
    if wandb_mode == 'n' and run_name is None:
        run_name = datetime.datetime.today().strftime('%Y%m%d%H%M')
    must_continue = False
    project = 'sfr_gallearn'
    if run_name is not None and os.path.isdir(
                os.path.join(
                    config.config[f'{__package__}_paths']['project_data_dir'],
                    run_name
                )
            ):
        model = ResNet(run_name).to(device)
        if wandb_mode == 'r':
            run_id = load_wandb_id(run_name)
            wandb.init(
                project=project,
                id=run_id,
                resume='must'
            )
    elif wandb_mode == 'r':
        # Exception if
        #     - run_name is None and wandb_mode is resume.
        #     - run_name is specified and wandb_mode is resume, but there is no 
        #       state file for it.
        # The argparser should take care of ensuring that the user provides a
        # run_name when wandb_mode is resume, so the following error message
        # will warn only about the state file.
        raise Exception(
            '`wandb_mode` is set to resume, but there is no corresponding'
            ' state file'
        )
    else:
        # If the state file doesn't exist

        # Things wandb will track
        #lr = 3.e-5 # learning rate
        lr = 1.e-3 # learning rate
        momentum = 0.5
        dataset = 'gallearn_data_256x256_3proj_wsat_wvmap_avg_sfr_tgt.h5'
        #dataset = 'ellipses.h5'
        n_blocks_list = [1, 1, 1, 1]
        out_channels_list = [16, 32, 64, 128]
        resblock = BasicResBlock 

        # Other things
        N_out_channels = 1

        if wandb_mode == 'y':
            wandb.init(
                # Set the wandb project where this run will be logged.
                project=project,

                # Track hyperparameters and run metadata.
                config={
                    "learning_rate": lr,
                    'momentum': momentum,
                    "dataset": dataset,
                    'batches': N_batches,
                    'n_blocks_list': n_blocks_list
                }
            )
            run_name = wandb.run.name

        # Define the model if we didn't rebuild one from a argument and
        # state files. Model is created on CPU by default so lazy layers can
        # be initialized before moving to MPS.
        model = ResNet(
                run_name,
                N_out_channels,
                lr,
                momentum,
                resblock,
                n_blocks_list,
                dataset,
                out_channels_list=out_channels_list,
                N_img_channels=4
            )
        #model = Net(
        #    kernel_size=3,
        #    conv_channels=[4, 8, 16, 32],
        #    N_groups=4,
        #    N_out_channels=1,
        #    lr=lr,
        #    momentum=momentum,
        #    run_name=run_name,
        #    dataset=dataset,
        #    scaling_function=preprocessing.std_asinh
        #)

        model.save_args()
        if wandb_mode == 'y':
            # Must wait until after we initialize the model to save the id
            # because the folder doesn't exist until after the model is built.
            save_wandb_id(wandb)

        must_continue = True
    
    ###########################################################################
    # Load the data
    ###########################################################################
    d = preprocessing.load_data(model.dataset)
    X = d['X'].to(device=device_str)
    X = model.scaling_function(X)[:Nfiles]
    ys = d['ys_sorted'].to(device=device_str)[:Nfiles]
    ys, means, stds = preprocessing.std_asinh(ys, 1.e11, return_distrib=True)
    rs = get_radii(d).to(device=device_str)

    N_all = len(ys) 
    print('{0:0.0f} galaxies in data'.format(N_all))
    N_test = max(1, int(0.15 * N_all))
    N_train = N_all - N_test

    # Train-test split
    idxs = torch.randperm(N_all, device=device_str)
    idxs_train, idxs_test = idxs[:N_train], idxs[N_train:]
    assert np.array_equal(
        np.arange(N_all),
        np.sort(np.concatenate((idxs_train, idxs_test)))
    ), (
        'There are indices missing from the train, test split, or there'
        ' are duplicates.'
    )

    ys_train = torch.index_select(ys, 0, idxs_train)
    ys_test = torch.index_select(ys, 0, idxs_test)
    X_train = torch.index_select(X, 0, idxs_train)
    X_test = torch.index_select(X, 0, idxs_test)
    rs_train = torch.index_select(rs, 0, idxs_train)
    rs_test = torch.index_select(rs, 0, idxs_test)
    ###########################################################################

    if must_continue:
        # Run a dummy fwd pass to initialize any lazy layers.
        # Must run on CPU first because MPS doesn't support lazy layer
        # materialization.
        model(X[:2].cpu(), rs[:2].cpu())
        model.apply(weights_init)  # Init model weights while still on CPU.
        model.to(device)
        model.init_optimizer()
    
        if wandb_mode == 'y':
            wandb.config['architecture'] = repr(model)

    # Learning rate scheduler that will make the new lr = `factor` * lr when 
    # it's been
    # `patience` epochs since the MSE less decreased by less than `threshold`.
    # I determined `threshold` by figuring I want the sqrt(MSE) to drop to 
    # ~0.045 when the sqrt(MSE) is 0.05. 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        model.optimizer,
        'min',
        factor=0.2,
        patience=7,
        threshold=3.e-4,
        threshold_mode='abs'
    )

    print(model)
     
    ###########################################################################
    # Make the DataLoaders.
    ###########################################################################
    batch_size_train = max(1, int(N_train / N_batches))
    batch_size_test = min(N_batches, N_test)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, rs_train, ys_train),
        batch_size=batch_size_train, 
        shuffle=True,
        generator=torch.Generator(device='cpu')
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test, rs_test, ys_test),
        batch_size=batch_size_test,
        shuffle=True,
        generator=torch.Generator(device='cpu')
    )
    ###########################################################################

    print('\nGetting initial performance.')
    test()

    print('Training.')
    for epoch in range(model.last_epoch + 1, model.last_epoch + N_epochs + 1):
        train_loss = train(epoch)
        test_loss = test()
        if wandb_mode in ['y', 'r']:
            wandb.log({
                'training loss': train_loss,
                'test loss': test_loss,
                'learning rate': model.optimizer.param_groups[0]['lr']
            })
        model.save_state(epoch, train_loss, test_loss)
        #scheduler.step(train_loss)

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
