import cnn
import preprocessing
import torch
import paths
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

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

i = 2 
d = preprocessing.load_data('gallearn_data_256x256_2d_tgt.h5')
X_all = d['X'].to(device=device_str)
X = preprocessing.new_min_max_scale(X_all)[i:i+1]
ys = d['ys_sorted'].to(device=device_str)[i:i+1]

run_name = 'visionary-darkness-190'
model = cnn.load_net(run_name)

def show_filters():
    results_dir = os.path.join(paths.data, run_name)
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    os.chdir(results_dir)
    
    model.eval()
    model.register_feature_hooks()

    model(X_all)
    pre_drop_feats = (
        model.features['conv_block']['14:ReLU']
        .flatten().detach().cpu().numpy()
    )
    plt.hist(pre_drop_feats, bins=50)
    plt.savefig('pre_drop_hist.png')
    plt.close()

    #post_drop_feats = (
    #    model.features['dropout:Dropout1d'].flatten().detach().cpu().numpy()
    #)
    #plt.hist(post_drop_feats, bins=50)
    #plt.savefig('post_drop_hist.png')
    #plt.close()


    ys_pred = model(X)
    print(ys_pred)
    last_conv_features = (
        model.features['conv_block']['12:Conv2d']
        .detach().cpu().numpy()[0]
    )
    three_conv_feats = (
        model.features['conv_block']['9:Conv2d']
        .detach().cpu().numpy()[0]
    )
    fifty_conv_feats = (
        model.features['conv_block']['0:Conv2d']
        .detach().cpu().numpy()[0]
    )

    fig = plt.figure(figsize=(3 * 10, 3 * 5))
    for i in range(50):
        ax = fig.add_subplot(5, 10, i+1)
        ax.imshow(
            fifty_conv_feats[i],
            cmap='viridis',
            interpolation='none'
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(plt.cm.viridis(0))
    fig.subplots_adjust(hspace=0., wspace=0.)
    plt.savefig('fifty_conv_features.png')
    plt.close()

    fig = plt.figure(figsize=(3 * 10, 3 * 5))
    for i in range(3):
        ax = fig.add_subplot(1, 3, i+1)
        ax.imshow(
            three_conv_feats[i],
            cmap='viridis',
            interpolation='none'
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(plt.cm.viridis(0))
    fig.subplots_adjust(hspace=0., wspace=0.)
    plt.savefig('three_conv_features.png')
    plt.close()

    fig = plt.figure(figsize=(3 * 10, 3 * 5))
    for i in range(1):
        ax = fig.add_subplot(1, 1, i+1)
        ax.imshow(
            last_conv_features[i],
            cmap='viridis',
            interpolation='none'
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(plt.cm.viridis(0))
    fig.subplots_adjust(hspace=0., wspace=0.)
    plt.savefig('last_conv_features.png')
    plt.close()

    plt.imshow(
        X[0][0].detach().cpu().numpy(),
        cmap='grey',
        interpolation='none',
        #norm=mpl.colors.LogNorm(vmin=5.e5, vmax=1.e8)
    )
    plt.savefig('gal.png')
    plt.close()

if __name__ == '__main__':
    show_filters()
