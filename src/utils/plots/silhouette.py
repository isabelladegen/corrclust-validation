import numpy as np
from matplotlib import pyplot as plt, cm

from src.utils.plots.matplotlib_helper_functions import reset_matplotlib, Backends


def plot_silhouette_analysis(silhouette_scores: {}, silhouette_avg: float, cluster_avgs: {} = None, title: str = "",
                             min_x=-0.5, backend:str=Backends.none.value, only_plot_clusters: [] = []):
    """Plots silhouette score for all clusters with more than one TS
    :param only_plot_clusters: if empty list plot all, otherwise plot only the clusters in the list
    :param cluster_avgs: average sil score for each cluster
    :param min_x: how much of the negative x axis to show
    :param silhouette_scores: dictionary {int, []} with cluster keys (int) and value being list of silhouette values for that cluster
    :param silhouette_avg: average silhouette score
    :param title: optional title for plot
    """
    label_font_size = 20
    k = len(silhouette_scores.keys())
    number_of_ts = sum([len(li) for li in silhouette_scores.values()])

    # configure plot
    gap = 25
    min_x = min_x
    reset_matplotlib(backend)

    fig, axs = plt.subplots(nrows=1,
                            ncols=1,
                            sharey=True,
                            sharex=True,
                            figsize=(15, 10), squeeze=0)
    ax = axs[0, 0]

    if title:
        ax.set_title(title, fontsize=label_font_size, y=1.05)

    ax.set_xlim([min_x, 1])
    # The (n_clusters+1)*gap is for inserting blank space between silhouette
    # plots of individual clusters
    ax.set_ylim([0, number_of_ts + (k + 1) * gap])

    # plot all silhouettes for non single ts clusters
    y_lower = gap
    clusters = silhouette_scores.keys()
    if len(only_plot_clusters) != 0:
        clusters = only_plot_clusters
    for cluster in clusters:
        ith_cluster_silhouette_values = silhouette_scores[cluster]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = len(ith_cluster_silhouette_values)
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(cluster) / k)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        cluster_label = "Cluster " + str(cluster) + ": " + str(size_cluster_i) + " items"
        # add cluster avg
        if cluster_avgs is not None:
            cluster_label = cluster_label + "; avg " + str(cluster_avgs[cluster])

        ax.text(-0.05, y_upper + 0.5, cluster_label, fontsize=label_font_size)

        # Compute the new y_lower for next plot
        y_lower = y_upper + gap  # 10 for the 0 samples

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2)

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([min_x, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.tick_params(axis='x', labelsize=label_font_size)
    ax.tick_params(axis='y', labelsize=label_font_size)
    ax.set_xlabel("Silhouette coefficient values", fontsize=label_font_size)
    ax.set_ylabel("Clusters", fontsize=label_font_size)
    fig.tight_layout()
    plt.show()
    return fig
