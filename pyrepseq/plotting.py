import string
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import scipy.cluster.hierarchy as hc
from scipy.interpolate import interpn

import logomaker as lm

import subprocess
from io import StringIO
from Bio import SeqIO


from .distance import *


def rankfrequency(
    data,
    ax=None,
    normalize_x=True,
    normalize_y=False,
    log_x=True,
    log_y=True,
    scalex=1.0,
    scaley=1.0,
    **kwargs,
):
    """
    Plot rank frequency plots.

    Parameters
    ----------
    data: array-like
        count data
    ax: `matplotlib.Axes`
        axes on which to plot the data
    normalize_x: bool, default:True
        whether to normalize counts to relative frequencies
    normalize_y: bool, default:False
        whether to normalize ranks to cumulative probabilities

    Returns
    -------
    list of `Line2D`
        Objectes representing the plotted data.
    """
    if ax is None:
        ax = plt.gca()
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    if normalize_x:
        data = data / np.sum(data)
    sorted_data = np.sort(data)
    # Cumulative counts:
    if normalize_y:
        norm = sorted_data.size
    else:
        norm = 1
    ret = ax.step(
        sorted_data[::-1] * scalex,
        scaley * np.arange(sorted_data.size) / norm,
        **kwargs,
    )
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    if normalize_x:
        ax.set_xlabel("Clone frequency")
    else:
        ax.set_xlabel("Clone size")
    if not normalize_y:
        ax.set_ylabel("Clone size rank")
    return ret


def labels_to_colors_hls(labels, palette_kws=dict(l=0.5, s=0.8), min_count=None):
    """
    Map a list of labels to a list of unique colors.
    Uses `seaborn.hls_palette`.

    Parameters
    ----------
    df : pandas DataFrame with data
    labels: list of labels
    min_count: map all labels seen less than min_count to black
    palette_kws: passed to `seaborn.hls_palette`
    """
    label, count = np.unique(labels, return_counts=True)
    if not min_count is None:
        label = label[count >= min_count]
    np.random.shuffle(label)
    lut = dict(zip(label, sns.hls_palette(len(label), **palette_kws)))
    return [lut[n] if n in lut else [0, 0, 0] for n in labels]


def labels_to_colors_tableau(labels, min_count=None):
    """
    Map a list of labels to a list of unique colors.
    Uses Tableau_10 colors

    Parameters
    ----------
    df : pandas DataFrame with data
    labels: list of labels
    min_count: map all labels seen less than min_count to black
    """
    label, count = np.unique(labels, return_counts=True)
    if not min_count is None:
        label = label[count >= min_count]
    # count, label = zip(*sorted(zip(count, label), reverse=True))
    np.random.shuffle(label)
    # cycler generator instantiation allows infinite sampling
    c = list(plt.cm.tab20.colors[::2])
    c.extend(plt.cm.tab20.colors[1::2])
    lut = dict(zip(label, plt.cycler(c=c)()))
    return [lut[n]["c"] if n in lut else [0, 0, 0] for n in labels]


class ClusterGridSplit(sns.matrix.ClusterGrid):
    """
    ClusterGrid subclass that provides separate data for upper and lower diagonal.
    """

    def __init__(self, data_lower, data_upper, **kws):
        super().__init__(data_lower, **kws)
        self.data_lower = data_lower
        self.data_upper = data_upper

    def plot_matrix(self, cbar_kws, xind, yind, **kws):
        self.data2d = np.tril(self.data_lower.iloc[yind, xind]) + np.triu(
            self.data_upper.iloc[yind, xind]
        )

        self.mask = self.mask.iloc[yind, xind]

        # Try to reorganize specified tick labels, if provided
        xtl = kws.pop("xticklabels", "auto")
        try:
            xtl = np.asarray(xtl)[xind]
        except (TypeError, IndexError):
            pass
        ytl = kws.pop("yticklabels", "auto")
        try:
            ytl = np.asarray(ytl)[yind]
        except (TypeError, IndexError):
            pass

        # Reorganize the annotations to match the heatmap
        annot = kws.pop("annot", None)
        if annot is None or annot is False:
            pass
        else:
            if isinstance(annot, bool):
                annot_data = self.data2d
            else:
                annot_data = np.asarray(annot)
                if annot_data.shape != self.data2d.shape:
                    err = "`data` and `annot` must have same shape."
                    raise ValueError(err)
                annot_data = annot_data[yind][:, xind]
            annot = annot_data

        # Setting ax_cbar=None in clustermap call implies no colorbar
        kws.setdefault("cbar", self.ax_cbar is not None)
        sns.matrix.heatmap(
            self.data2d,
            ax=self.ax_heatmap,
            cbar_ax=self.ax_cbar,
            cbar_kws=cbar_kws,
            mask=self.mask,
            xticklabels=xtl,
            yticklabels=ytl,
            annot=annot,
            **kws,
        )

        ytl = self.ax_heatmap.get_yticklabels()
        ytl_rot = None if not ytl else ytl[0].get_rotation()
        self.ax_heatmap.yaxis.set_ticks_position("right")
        self.ax_heatmap.yaxis.set_label_position("right")
        if ytl_rot is not None:
            ytl = self.ax_heatmap.get_yticklabels()
            plt.setp(ytl, rotation=ytl_rot)

        tight_params = dict(h_pad=0.02, w_pad=0.02)
        if self.ax_cbar is None:
            self.tight_layout(**tight_params)
        else:
            # Turn the colorbar axes off for tight layout so that its
            # ticks don't interfere with the rest of the plot layout.
            # Then move it.
            self.ax_cbar.set_axis_off()
            self.fig.tight_layout(**tight_params)
            self.ax_cbar.set_axis_on()
            self.ax_cbar.set_position(self.cbar_pos)


def clustermap_split(
    data_lower,
    data_upper,
    *,
    pivot_kws=None,
    method="average",
    metric="euclidean",
    z_score=None,
    standard_scale=None,
    figsize=(10, 10),
    cbar_kws=None,
    row_cluster=True,
    col_cluster=True,
    row_linkage=None,
    col_linkage=None,
    row_colors=None,
    col_colors=None,
    mask=None,
    dendrogram_ratio=0.2,
    colors_ratio=0.03,
    cbar_pos=(0.02, 0.8, 0.05, 0.18),
    tree_kws=None,
    **kws,
):
    """
    Convenience function for instantiating a `ClusterGridSplit` instance and calling the plot routine.
    """
    plotter = ClusterGridSplit(
        data_lower,
        data_upper,
        pivot_kws=pivot_kws,
        figsize=figsize,
        row_colors=row_colors,
        col_colors=col_colors,
        z_score=z_score,
        standard_scale=standard_scale,
        mask=mask,
        dendrogram_ratio=dendrogram_ratio,
        colors_ratio=colors_ratio,
        cbar_pos=cbar_pos,
    )

    return plotter.plot(
        metric=metric,
        method=method,
        colorbar_kws=cbar_kws,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        row_linkage=row_linkage,
        col_linkage=col_linkage,
        tree_kws=tree_kws,
        **kws,
    )


def similarity_clustermap(
    df,
    alpha_column="cdr3a",
    beta_column="cdr3b",
    norm=None,
    bounds=np.arange(0, 7, 1),
    linkage_kws=dict(method="average", optimal_ordering=True),
    cluster_kws=dict(t=6, criterion="distance"),
    cbar_kws=dict(label="Sequence Distance", format="%d", orientation="horizontal"),
    meta_columns=None,
    meta_to_colors=None,
    **kws,
):
    """
    Plots a sequence-similarity clustermap.

    Parameters
    ----------
    df : pandas DataFrame with data
    alpha_column, beta_column: column name with alpha and beta amino acid information (set one to None for single chain plotting)
    norm: `matplotlib.colors.Normalize` subclass for turning distances into colors
    bounds: bounds used for colormap `matplotlib.colors.BoundaryNorm` (only used if norm = None)
    linkage_kws: keyword arguments for linkage algorithm
    cluster_kws: keyword arguments for clustering algorithm
    cbar_kws: keyword arguments for colorbar
    meta_columns: list-like
        metadata to plot alongside the cluster assignment
    meta_to_colors: list-like
        list of functions mapping metadata labels to colors
        first element of list is for clusters
    kws: keyword arguments passed on to the clustermap.

    """

    if meta_to_colors is None:
        if meta_columns is None:
            meta_to_colors = [labels_to_colors_hls]
        else:
            meta_to_colors = [labels_to_colors_hls] * (len(meta_columns) + 1)

    is_single_chain = (alpha_column is None) or (beta_column is None)
    if is_single_chain: 
        chain = beta_column if alpha_column is None else alpha_column
        sequences = df[chain]
        distances = pdist(sequences)
    else:
        sequences_alpha = df[alpha_column]
        sequences_beta = df[beta_column]
        sequences = sequences_alpha + "_" + sequences_beta
        distances_alpha = pdist(sequences_alpha)
        distances_beta = pdist(sequences_beta)
        distances = distances_alpha + distances_beta

    linkage = hc.linkage(distances, **linkage_kws)
    cluster = hc.fcluster(linkage, **cluster_kws)

    cmap = plt.cm.viridis
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "Custom cmap", list(reversed(cmaplist)), cmap.N
    )

    if norm is None:
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        # plot tick in the middle of the discretized colormap
        cbar_kws.update(dict(ticks=bounds[:-1] + 0.5))

    cluster_colors = pd.Series(meta_to_colors[0](cluster, min_count=2), name="Cluster")
    if not meta_columns is None:
        colors_list = [cluster_colors]
        if type(meta_columns) is dict:
            meta_colors = [
                pd.Series(mapper(df[col]), name=meta_columns[col])
                for col, mapper in zip(meta_columns, meta_to_colors[1:])
            ]
        else:
            meta_colors = [
                pd.Series(mapper(df[col]), name=col)
                for col, mapper in zip(meta_columns, meta_to_colors[1:])
            ]
        colors_list.extend(meta_colors)
        colors = pd.concat(colors_list, axis=1)
    else:
        colors = cluster_colors

    # default clustermap kws
    clustermap_kws = dict(
        cbar_kws=cbar_kws,
        dendrogram_ratio=0.12,
        colors_ratio=0.04,
        cbar_pos=(0.38, 0.99, 0.4, 0.02),
        rasterized=True,
        figsize=(4.2, 4.2),
        xticklabels=[],
        yticklabels=[],
    )
    clustermap_kws.update(kws)

    if is_single_chain:
        # For plotting purposes to plot upper and lower diagonal with same info
        distances_alpha = distances
        distances_beta = distances

    cg = clustermap_split(
        pd.DataFrame(squareform(distances_alpha)),
        pd.DataFrame(squareform(distances_beta)),
        row_linkage=linkage,
        col_linkage=linkage,
        cmap=cmap,
        norm=norm,
        row_colors=colors,
        **clustermap_kws,
    )

    if norm is None:
        cbar_labels = [str(b) for b in bounds[:-1]]
        cbar_labels[-1] = ">" + cbar_labels[-1]
        cg.cax.set_xticklabels(cbar_labels)
    if is_single_chain:
        label = r"CDR3$\beta$ Sequence" if 'b' in chain else r"CDR3$\alpha$ Sequence" 
        cg.ax_heatmap.set_xlabel(label)
        cg.ax_heatmap.set_ylabel(label)
    else:
        cg.ax_heatmap.set_xlabel(r"CDR3$\alpha$ Sequence")
        cg.ax_heatmap.set_ylabel(r"CDR3$\beta$ Sequence")
    cg.ax_col_dendrogram.set_visible(False)
    return cg, linkage, cluster


def label_axes(
    fig_or_axes,
    labels=string.ascii_uppercase,
    labelstyle=r"%s",
    xy=(-0.1, 0.95),
    xycoords="axes fraction",
    **kwargs,
):
    """
    Walks through axes and labels each.
    kwargs are collected and passed to `annotate`

    Parameters
    ----------
    fig : Figure or Axes to work on
    labels : iterable or None
        iterable of strings to use to label the axes.
        If None, lower case letters are used.

    loc : Where to put the label units (len=2 tuple of floats)
    xycoords : loc relative to axes, figure, etc.
    kwargs : to be passed to annotate
    """
    # re-use labels rather than stop labeling
    annotate_kwargs = dict(fontweight="bold", va="top")
    annotate_kwargs.update(kwargs)
    labels = itertools.cycle(labels)
    axes = fig_or_axes.axes if isinstance(fig_or_axes, plt.Figure) else fig_or_axes
    for ax, label in zip(axes, labels):
        ax.annotate(labelstyle % label, xy=xy, xycoords=xycoords, **annotate_kwargs)


def align_seqs(seqs):
    """Align multiple sequences using mafft-linsi with default parameters.

    Parameters
    ----------
    seqs: iterable of strings

    Returns
    -------
    list of strings
        aligned sequences (with gaps)
    """
    seq_str = ""
    for i, seq in enumerate(seqs):
        seq_str += f"> seq {i}\n"
        seq_str += f"{seq}\n"
    child = subprocess.Popen(
        ["mafft-linsi", "--quiet", "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    child.stdin.write(seq_str.encode())
    child_out = child.communicate()[0].decode("utf8")
    seqs_aligned = list(SeqIO.parse(StringIO(child_out), "fasta"))
    child.stdin.close()
    return [str(seq.seq) for seq in seqs_aligned]


def seqlogos(seqs, ax=None, **kwargs):
    """
    Display a sequence logo.

    Aligns sequences using `align_seqs` if they are are not of equal length.

    Parameters
    ----------
    seqs: iterable of strings
        sequences to be displayed
    ax: matplotlib.axes
        if None create new figure
    **kwargs: dict
        passed on to logomaker.Logo

    Returns
    -------
    axes, counts_matrix
    """
    lengths = np.array([len(s) for s in seqs])
    if len(np.unique(lengths)) > 1:
        seqs = align_seqs(seqs)
    counts_mat = lm.alignment_to_matrix(seqs)
    if ax is None:
        fig, ax = plt.subplots(figsize=(0.3 * counts_mat.shape[0], 0.4))
    lm_kwargs = dict(color_scheme="chemistry", show_spines=False, baseline_width=0.0)
    lm_kwargs.update(kwargs)
    lm.Logo(counts_mat, ax=ax, **lm_kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["bottom"].set_visible(False)
    if ax is None:
        fig.tight_layout()
    return ax, counts_mat


def seqlogos_vj(df, cdr3_column, v_column, j_column, axes=None, **kwargs):
    """
    Display a sequence logo with V and J gene information.

    Parameters
    ----------
    df: pd.DataFrame
        input data
    cdr3_column: str
        column name for cdr3 sequences
    v_column: str
        column name for v genes
    j_column: str
        column name for j genes
    **kwargs: dict
        passed on to `seqlogos`
    """
    seqs = df[cdr3_column]
    max_length = max([len(s) for s in seqs])
    counts_v = df[v_column].value_counts()
    counts_j = df[j_column].value_counts()

    if axes is None:
        fig, axes = plt.subplots(
            figsize=(0.2 * max_length + 2.0, 0.4),
            ncols=3,
            gridspec_kw=dict(width_ratios=(1, 0.2 * max_length, 1), wspace=0.1),
            sharey=True,
        )
    seqlogos(seqs, axes[1], **kwargs)
    for ax, counts in zip([axes[0], axes[2]], [counts_v, counts_j]):
        previous_count = 0
        for gene_name, cum_count in counts.cumsum().items():
            lm.Glyph(0.5, gene_name[3:], previous_count, cum_count, ax=ax)
            previous_count = cum_count
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
    return axes


class HandlerTupleOffset(mpl.legend_handler.HandlerTuple):
    """
    Legend Handler for tuple plotting markers on top of each other
    """

    def __init__(self, horizontal=True, **kwargs):
        """
        horizontal: shift horizontally (for markers), else shift vertically (for lines)
        """
        self.horizontal = horizontal
        mpl.legend_handler.HandlerTuple.__init__(self, **kwargs)

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        horizontal = self.horizontal
        nhandles = len(orig_handle)
        perside = (nhandles - 1) / 2
        offset = (width if horizontal else height) / nhandles
        handler_map = legend.get_legend_handler_map()
        a_list = []
        for i, handle1 in enumerate(orig_handle):
            handler = legend.get_legend_handler(handler_map, handle1)
            _a_list = handler.create_artists(
                legend,
                handle1,
                xdescent + (offset * (i - perside) if horizontal else 0),
                ydescent + (offset * (i - perside) if not horizontal else 0),
                width,
                height,
                fontsize,
                trans,
            )
            a_list.extend(_a_list)
        return a_list


def density_scatter(
    x, y, ax=None, discrete=False, sort=True, bins=20, trans=None, **kwargs
):
    """
    Scatter plot with color indicating point density estimated by local binning.

    ax: matplotlib.Axes
        axes on which to plot
    discrete: Boolean
        Is the data discrete? -> count-based density
    bins: int
        number of bins for density estimation
    trans: function
       transformation to apply before density estimation
    sort: Boolean
        sort the data points by density to plot densest points last.
    **kwargs:
        passed on to ax.scatter

    """
    x = np.asarray(x)
    y = np.asarray(y)
    if ax is None:
        ax = plt.gca()

    if discrete:
        points, counts = np.unique(
            np.array(list(zip(x, y))), return_counts=True, axis=0
        )
        x = points[:, 0]
        y = points[:, 1]
        z = counts
    else:
        if trans is None:
            trans = lambda x: x
        data, x_e, y_e = np.histogram2d(trans(x), trans(y), bins=bins)
        z = interpn(
            (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
            data,
            np.vstack([trans(x), trans(y)]).T,
            method="splinef2d",
            bounds_error=False,
        )

    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter(x, y, c=z, **kwargs)
    return ax
