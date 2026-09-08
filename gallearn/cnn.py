import torch.nn as nn
import torch
import datetime


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

        super().__init__()

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
        x = nn.functional.dropout(x, 0.2, training=self.training)
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


def _replace_first_conv_in_channels(
        module, in_channels, preserve_weights=False):
    """
    Recursively find the first Conv2d layer in a module and replace it
    with a new Conv2d that has the specified in_channels.
    Returns True if a replacement was made, False otherwise.

    preserve_weights : bool
        If True, copy the old conv's weights into the first
        min(old_in_channels, in_channels) channels of the new conv
        instead of leaving it randomly initialized, so a pretrained
        backbone keeps its pretrained conv1 filters for the channels
        it already had (e.g. the 3 image bands), and only the extra
        channels (e.g. a velocity map) start from scratch. Extra
        channels are initialized to the mean of the old conv's
        weights across its input channels, rather than a fresh
        Kaiming init, so their initial output scale roughly matches
        the pretrained channels' instead of starting uncorrelated
        with everything the rest of the pretrained network expects.
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
            if preserve_weights:
                old_in_channels = child.in_channels
                n_copy = min(old_in_channels, in_channels)
                with torch.no_grad():
                    new_conv.weight[:, :n_copy] = (
                        child.weight[:, :n_copy]
                    )
                    if in_channels > old_in_channels:
                        mean_weight = child.weight.mean(
                            dim=1, keepdim=True
                        )
                        new_conv.weight[:, old_in_channels:] = (
                            mean_weight
                        )
                    if child.bias is not None:
                        new_conv.bias[:] = child.bias
            setattr(module, name, new_conv)
            return True
        if _replace_first_conv_in_channels(
                child, in_channels, preserve_weights=preserve_weights):
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


class StandardNet(nn.Module):
    def __init__(
                self,
                lr,
                momentum,
                backbone,
                dataset,
                in_channels=None,
                pretrained=False,
            ):
        from . import preprocessing

        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.backbone = backbone
        self.dataset = dataset

        _make_backbone_feature_extractor(self.backbone)

        if in_channels is not None:
            # pretrained backbones ship conv1 weights fit to their
            # original 3-channel input; preserve those in the new
            # conv rather than discarding them, so pretraining still
            # helps the very first layer instead of only layer1+.
            if not _replace_first_conv_in_channels(
                    self.backbone, in_channels,
                    preserve_weights=pretrained):
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
            weight_decay=1e-4,
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
                N_img_channels=None,
                head_widths=None):

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
            head_widths: hidden layer widths for the fully-connected
                head, applied after global average pooling. Required;
                there is no default, so no head shape is assumed
                silently. See gallearn.train.HEAD_PRESETS for the
                named presets docs/status.md's head-width sweep
                tested.
        '''
        from . import config
        import os
        from . import preprocessing

        super().__init__()

        self.run_name = run_name

        self.run_dir = os.path.join(
            config.config[f'{__package__}_paths']['project_data_dir'],
            run_name
        )

        self.resblock = resblock
        self.N_out_channels = N_out_channels
        self.momentum = momentum
        self.lr = lr
        self.n_blocks_list = n_blocks_list
        self.out_channels_list = out_channels_list
        self.N_img_channels = N_img_channels
        self.head_widths = head_widths
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

        # Head. self.head_widths gives the hidden-layer widths; the
        # first Linear is Lazy since the pooled-feature width depends
        # on out_channels_list and the concatenated auxiliary inputs
        # (see forward()), not on head_widths itself.
        head_layers = []
        prev_width = None
        for i, width in enumerate(self.head_widths):
            if i == 0:
                head_layers.append(nn.LazyLinear(width))
            else:
                head_layers.append(nn.Linear(prev_width, width))
            head_layers.append(nn.BatchNorm1d(width))
            head_layers.append(self.activation_module())
            prev_width = width
        head_layers.append(nn.Linear(prev_width, self.N_out_channels))
        self.head = nn.Sequential(*head_layers)

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
        x = torch.nn.functional.dropout(
            x,
            0.5,
            training=self.training,
        )
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
