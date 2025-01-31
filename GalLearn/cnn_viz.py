import cnn
import preprocessing
import torch

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

i = 8 
d = preprocessing.load_data('gallearn_data_256x256_2d_tgt.h5')
X = d['X'].to(device=device_str)
X = preprocessing.new_min_max_scale(X)[i:i+1]
ys = d['ys_sorted'].to(device=device_str)[i:i+1]

model = cnn.load_net('trim-bird-186')

def show_filters():
    model.eval()
    model.register_feature_hooks()
    ys_pred = model(X)
    filters = (
        model.features['conv_block']['12:Conv2d'].detach().cpu().numpy()[0]
    )

    fig = plt.figure(figsize=(3 * 10, 3 * 5))

    print(ys_pred)
    for i in range(1):
        ax = fig.add_subplot(1, 1, i+1)
        ax.imshow(
            filters[i],
            cmap='viridis',
            interpolation='none'
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(plt.cm.viridis(0))
    fig.subplots_adjust(hspace=0., wspace=0.)
    plt.savefig('filters.png')
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
