import torch
import cmasher as cmr

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['mathtext.fontset'] = 'dejavuserif'
rcParams['font.family'] = 'serif' 
#rcParams['xtick.labelsize'] = 16
#rcParams['ytick.labelsize'] = 16
#rcParams['axes.grid']=True
rcParams['axes.titlesize']=24
#rcParams['axes.labelsize']=20
rcParams['axes.titlepad']=15
rcParams['legend.frameon'] = True
rcParams['legend.facecolor']='white'
rcParams['figure.facecolor'] = (1., 1., 1., 1.) #white with alpha=1.

def embeddable_image(data):
    import base64
    from io import BytesIO
    from PIL import Image

    img_data = 255 - 15 * data.to(torch.uint8)
    image = Image.fromarray(img_data.detach().cpu().numpy(), mode='L')
    image = image.resize((64, 64), Image.Resampling.BICUBIC)
    buffer = BytesIO()
    image.save(buffer, format='png')
    for_encoding = buffer.getvalue()
    return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()

def make_embedding(data):
    import umap
    import preprocessing
    import torch

    reducer = umap.UMAP(n_neighbors=5)
    X = preprocessing.std_asinh(data['X'], stretch=1.e-5)
    X = torch.flatten(X, start_dim=1)
    reducer.fit(X)
    embedding = reducer.transform(X)

    return embedding

def plt_umap():
    import preprocessing

    data = preprocessing.load_data(
        'gallearn_data_128x128_3proj_wsat_sfr_tgt.h5'
    )
    embedding = make_embedding(data)
    sfrs = data['ys_sorted']
    sfrs = preprocessing.std_asinh(sfrs, 1.e11)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cmap = cmr.get_sub_cmap('viridis', 0., 0.9)
    pathcollection = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        marker='.',
        alpha=0.6,
        c=sfrs,
        cmap=cmap
    )
    ax.set_aspect('equal', 'datalim')
    ax.set_title(
        'Uniform Manifold Approximation and Projection\n3-Channel Mock Images'
    )

    cb = plt.colorbar(pathcollection, pad=0.05, location='right')
    cb.ax.tick_params(labelsize=12)
    cb.set_label(
        size=12,
        label=(
            '$\mathrm{asinh}'
            '\left(\\frac{SSFR}{\mathrm{yr}^{-1}}\\times 10^{11}\\right)$'
        )
    )

    plt.show()

    return None

def plt_interactive_umap():
    import preprocessing
    import umap
    import pandas as pd
    import numpy as np

    import bokeh
    from bokeh.plotting import figure, show, output_notebook

    output_notebook()

    d = preprocessing.load_data('gallearn_data_128x128_3proj_wsat_sfr_tgt.h5')
    images = preprocessing.std_asinh(d['X'], stretch=1.e-5)
    sfrs = d['ys_sorted']
    scaled_sfrs = preprocessing.std_asinh(sfrs, 1.e11)

    embedding = make_embedding(d)

    df = pd.DataFrame(embedding, columns=('x', 'y'))
    df['sfr'] = sfrs
    df['image'] = list(map(
        embeddable_image, 
        images[:, 0]
    ))

    colors = [
        "#%02x%02x%02x" % (int(r), int(g), int(b)) 
        for r, g, b, _ 
        in 255 * mpl.cm.viridis(mpl.colors.Normalize()(scaled_sfrs.flatten()))
    ]
    df['marker_color'] = colors

    datasource = bokeh.models.ColumnDataSource(df)
    plot_figure = figure(
        title=(
            'Uniform Manifold Approximation and Projection'
            '\n3-Channel Mock Images'
        ),
        width=600,
        height=600,
        tools=('pan, wheel_zoom, reset')
    )

    plot_figure.add_tools(bokeh.models.HoverTool(tooltips="""
       <div>
           <div>
               <img src='@image', style='float left; margin: 5px 5px 5px 5px'/>
           </div>
           <div>
               <span style='font-size: 16px; color: #224499'>SFR:</span>
               <span style='font-size: 18px'>@sfr</span>
           </div>
       </div>
       """
    ))

    plot_figure.scatter(
        'x',
        'y',
        source=datasource,
        fill_color='marker_color',
        line_alpha=0.6,
        fill_alpha=0.6,
        size=4
    )
    show(plot_figure)
