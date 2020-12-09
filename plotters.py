import itertools
from pathlib import Path

from matplotlib import pyplot as plt


def plot_dataset_to(dataset, path, dataset_pos=None, xlim=None, ylim=None, verbose=True):
    if dataset.ndim != 2:
        raise TypeError("'dataset' must be a 2d array")

    path = Path(path)
    fig, ax = plt.subplots()
    for igl, gl in enumerate(dataset):
        if verbose:
            print("Plotting plot %04d..." % igl, flush=True, end="\r")
        ax.clear()
        ax.plot(gl)
        if dataset_pos is not None:
            p0, p1 = dataset_pos[igl]
        else:
            p0, p1 = 0, dataset.shape[1]
        ax.axvline(p0, c='red', ls='--')
        ax.axvline(p1, c='red', ls='--')
        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)
        n_igl = int(np.log10(len(dataset))) + 1
        filename = path + f"dataset_plot-{igl:0{n_igl}d}.png"
        fig.savefig(filename)
    plt.close(fig)


def confused_plot(ax, cmat, labels, ilabels):
    ax.imshow(cmat, cmap=plt.get_cmap('Blues'), vmin=0, vmax=cmat.sum()*0.2)
    for i, j in itertools.product(ilabels, repeat=2):
        ax.annotate(str(int(cmat[i,j])), xy=(j,i), ha='center', va='center')
    ax.set_xticks(ilabels)
    ax.set_yticks(ilabels[:-1])  # Extra tick for 2nd-level label
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels[:-1])
    ax.set_xlim([-0.5, len(labels)-0.5])
    ax.set_ylim([-0.5, len(labels)-1.5])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.grid(False)