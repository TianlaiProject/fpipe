import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec

def setup_axes(m=4, n=5, colorbar=True, title='', figsize=(15, 7)):
    
    fig = plt.figure(figsize=figsize)
    
    gs = gridspec.GridSpec(m, n, left=0.05, bottom=0.1, top=0.95, right=0.98,
            figure=fig, wspace=0.0, hspace=0.0)
    
    axes = []
    for bi in range(19):
        i = int(bi // n)
        j = int(bi - i * n)
        ax = fig.add_subplot(gs[i,j])
            
        if i != m-1:
            ax.set_xticklabels([])
        if j != 0:
            ax.set_yticklabels([])
                
                
        axes.append(ax)

    if colorbar:
        if m == 5:
            cax = fig.add_axes([0.85, 0.17, 0.19, 0.02])
        else:
            cax = fig.add_axes([0.84/float(m) * (m - 1) + 0.16, 0.14, 0.19, 0.02])
        axes.append(cax)
    else:
        if m == 5:
            cax = fig.add_axes([0.75, 0.17, 0.19, 0.02])
        else:
            cax = fig.add_axes([0.84/float(m) * (m - 1) + 0.16, 0.14, 0.19, 0.02])
        #cax.get_xaxis().set_visible(False)
        #cax.get_yaxis().set_visible(False)
        cax.set_axis_off()
        cax.set_title(title)
            
    return fig, axes

