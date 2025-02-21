import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.evaluation.distance_metric_assessment import DistanceMeasureCols
from src.evaluation.distance_metric_evaluation import inverse_criteria
from src.utils.distance_measures import short_distance_measure_names
from src.utils.plots.matplotlib_helper_functions import reset_matplotlib, Backends, fontsize

display_name_data_variant = {
    "complete, raw": "Raw, 100%",
    "complete, correlated": "Correlated, 100%",
    "complete, non-normal": "Non-normal, 100%",
    "complete, downsampled": "Downsampled, 100%",
    "partial, raw": "70%",
    "partial, correlated": "70%",
    "partial, non-normal": "70%",
    "partial, downsampled": "70%",
    "sparse, raw": "10%",
    "sparse, correlated": "10%",
    "sparse, non-normal": "10%",
    "sparse, downsampled": "10%",
}


def heatmap_of_raw_values(df: pd.DataFrame, highlight_rows=[], backend=Backends.none.value):
    """
    Creates heatmap of raw across data variants, for the colour the raw values are scaled, for annotation the actual
    raw values are used
    :param df: Dataframe with rows being data variants and columns being distance measures
    :param highlight_rows: sets all other rows transparent, if no rows or cols given then everything is as normal
    """
    # Setup plt
    reset_matplotlib(backend)
    fig = plt.figure(figsize=(16, 12))

    do_highlight = len(highlight_rows) > 0
    # Create masks for highlighted row and column
    row_mask = np.zeros_like(df, dtype=bool)

    # Set True for the rows to highlight
    for row in highlight_rows:
        row_mask[df.index == row] = True

    # Adjust alpha values
    alpha_matrix = np.ones_like(df, dtype=float)
    alpha_matrix[row_mask] = 1.0  # Highlighted cells fully opaque
    alpha_matrix[~row_mask] = 0.5 if do_highlight else 1.0  # Other cells more transparent unless no highlight

    # scale values so that all colors are comparable (dark is good, bright is worse)
    scaled_df = df.copy()
    measures = []
    criteria = []
    for col in df.columns:
        split_str = col.split(':')
        criteria.append(split_str[0])
        measures.append(split_str[1])

    m1 = measures[0]
    m2 = measures[1]

    for criterion in set(criteria):
        col1 = f'{criterion}:{m1}'
        col2 = f'{criterion}:{m2}'

        # Find min and max across both measures for this criterion
        min_val = min(df[col1].min(), df[col2].min())
        max_val = max(df[col1].max(), df[col2].max())

        # Scale both measures together, inverse for measures where lower values are better
        for col, measure in [(col1, m1), (col2, m2)]:
            scaled = (df[col] - min_val) / (max_val - min_val)
            scaled_df[col] = 1 - scaled if criterion in inverse_criteria else scaled

    # Create heatmap
    ax = sns.heatmap(
        data=scaled_df,  # show scaled values for colour comparison
        annot=df,  # Show raw values in cell annotation
        fmt='.2f',  # Format for annotations
        cmap=sns.color_palette("mako", n_colors=256, as_cmap=True).reversed(),
        cbar_kws={'label': 'Normalised raw values',
                  'shrink': 1,
                  'aspect': 30
                  },
        annot_kws={'size': fontsize, 'weight': 'bold'},
        square=True,
        alpha=alpha_matrix
    )
    ax.tick_params(left=False, bottom=False, top=False)
    ax.grid(False)

    # Set transparency of text
    def text_alpha(is_highlighted):
        if is_highlighted:
            return 1.0
        return 0.4 if do_highlight else 1.0

    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            text = ax.texts[i * len(df.columns) + j]
            text.set_alpha(text_alpha(row_mask[i, j]))

    # Separate each criterion pair
    n_cols = len(df.columns)
    for i in range(0, n_cols, 2):
        if i > 0:  # Don't draw line at x=0
            ax.axvline(x=i, color='#B0C4DE', linewidth=3)

    # Axis labels
    plt.yticks(rotation=0, ha='right', fontsize=fontsize)  # data variant
    plt.xticks(rotation=90, ha='right', fontsize=fontsize, ticks=np.arange(len(df.columns)) + 0.7, labels=measures)

    # Add criteria labels on top
    secondary_ax = plt.gca().secondary_xaxis('top')  # or 'bottom' if you prefer
    secondary_ax.set_xticks(np.arange(1, len(df.columns), 2))  # every other column
    secondary_ax.set_xticklabels(criteria[::2], fontsize=fontsize)
    secondary_ax.tick_params(axis='x', length=0)  # hide the actual tick marks
    secondary_ax.spines['top'].set_visible(False)  # hide spine that the tick marks are on

    plt.tight_layout()
    plt.show()

    return fig


def heatmap_of_ranks(df: pd.DataFrame, highlight_rows=[], highlight_cols=[], figsize=(16, 12),
                     backend=Backends.none.value):
    """
    Creates heatmap of rank across data variants
    :param df: Dataframe with rows being data variants and columns being distance measures
    :param highlight_cols: sets all other cols transparent
    :param highlight_rows: sets all other rows transparent, if no rows or cols given then everything is as normal
    """
    # Create figure and axes
    # Setup plt
    reset_matplotlib(backend)
    fig = plt.figure(figsize=figsize)

    do_highlight = (len(highlight_rows) + len(highlight_cols)) > 0

    # Create masks with trues for row and cols
    mask = np.zeros(df.shape, dtype=bool)
    row_indices = [df.index.get_loc(row) for row in highlight_rows]
    col_indices = [df.columns.get_loc(col) for col in highlight_cols]
    for r, c in zip(row_indices, col_indices):
        mask[r, c] = True

    # Set True for the row and column to highlight
    # for row in highlight_rows:
    #     row_mask[df.index == row] = True
    # for col in highlight_cols:
    #     col_mask[:, df.columns == col] = True

    # Adjust alpha values to be 1.0 for highlight and more transparent for others
    # mask = row_mask | col_mask
    alpha_matrix = np.ones_like(df, dtype=float)
    alpha_matrix[mask] = 1.0  # Highlighted cells fully opaque
    alpha_matrix[~mask] = 0.6 if do_highlight else 1.0  # Other cells more transparent unless we're not highlighting

    # translate labels
    translated_labels = [display_name_data_variant[idx] for idx in df.index]

    # Create heatmap
    ax = sns.heatmap(
        df,
        annot=True,  # Show values in cells
        fmt='.1f',  # Format for annotations
        cmap=sns.color_palette("mako", n_colors=256, as_cmap=True),
        cbar_kws={'label': 'Rank',
                  'shrink': 0.68,
                  'aspect': 30
                  },
        annot_kws={'size': fontsize, 'weight': 'bold'},
        square=True,
        linewidths=1,
        linecolor='black',
        clip_on=False,
        alpha=alpha_matrix,
        yticklabels=translated_labels
    )
    ax.tick_params(left=False, bottom=False)
    ax.grid(False)

    # Add pink borders for masked cells
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            if mask[i, j]:
                # Add a rectangle with pink edges around the cell
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                           edgecolor='#BFFF00',
                                           linewidth=2,
                                           clip_on=False))

    # Set transparency of text
    def text_alpha(is_highlighted):
        if is_highlighted:
            return 1.0
        return 0.8 if do_highlight else 1.0

    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            text = ax.texts[i * len(df.columns) + j]
            text.set_alpha(text_alpha(mask[i, j]))

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90, ha='right', fontsize=fontsize)
    plt.yticks(rotation=0, ha='right', fontsize=fontsize)

    plt.tight_layout()
    plt.show()

    return fig


def violin_plots_of_average_rank_per_distance_measure(df: pd.DataFrame, title="",
                                                      rename_measures=short_distance_measure_names,
                                                      backend=Backends.none.value):
    """Input df is rows are the runs, columns are the measures"""

    # Setup plt
    reset_matplotlib(backend)

    x_column = DistanceMeasureCols.type
    y_column = 'Average Rank'

    # reshape from x=run, y=measures to two columns measure and average rank
    melted_df = df.melt(var_name=x_column, value_name=y_column)
    melted_df[x_column] = melted_df[x_column].replace(rename_measures)
    min_rank = melted_df[y_column].min()

    # calculate median for each measure and sort
    measure_medians = melted_df.groupby(x_column)[y_column].median().sort_values()
    measure_order = measure_medians.index.tolist()

    fig = plt.figure(figsize=(12, 6))
    ax = sns.violinplot(data=melted_df,
                        x=x_column,
                        y=y_column,
                        inner='box',  # Shows quartile box inside violin
                        alpha=0.7,
                        order=measure_order,
                        bw_adjust=0.4)  # reduce the smoothing to better represent the discrete values

    # Add median annotations below the violins
    for i, median in enumerate(measure_medians):
        ax.text(i, int(min_rank), f'{median:.2f}',
                horizontalalignment='center',
                verticalalignment='center',
                fontweight='bold',
                fontsize=15,
                color='#404040')  # Dark grey color

    # dotted grid lines
    ax.yaxis.grid(True, linestyle=':', color='gray')

    # Customize the plot
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    if title != "":
        plt.title(title)
    plt.ylabel('Average Rank')
    plt.xlabel('')

    # Set y-axis to start at 1
    plt.ylim(bottom=int(min_rank) - 0.5)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Show the plot
    plt.show()
    return fig


def violin_plot_grids_per_criteria_for_distance_measure(df: pd.DataFrame, title="",
                                                        rename_measures=short_distance_measure_names,
                                                        backend=Backends.none.value):
    # Setup plt
    reset_matplotlib(backend)

    criteria = df[DistanceMeasureCols.criterion].unique()
    n_criteria = len(criteria)
    figsize = (18, 25)

    # Calculate grid dimensions
    n_rows = 3
    n_cols = 2
    x_column = DistanceMeasureCols.type
    y_column = DistanceMeasureCols.rank

    assert n_rows * n_cols >= len(criteria), "Not big enough grid for all criteria"

    # rename measures
    df = df.replace(rename_measures)

    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True)
    fig.suptitle(title, fontsize=fontsize, y=0.98)
    axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

    # Create violin plots for each criterion
    for idx, criterion in enumerate(criteria):
        # Filter data for current criterion
        criterion_data = df[df[DistanceMeasureCols.criterion] == criterion]

        # calculate median for each measure and sort
        measure_medians = criterion_data.groupby(x_column)[y_column].median().sort_values()
        measure_order = measure_medians.index.tolist()

        # Create violin plot
        ax = sns.violinplot(data=criterion_data,
                            x=x_column,
                            y=y_column,
                            ax=axes[idx],
                            inner='box',  # Shows quartile box inside violin
                            alpha=0.7,
                            order=measure_order,  # lowest median first
                            bw_adjust=0.4  # less smoothing to be truer to actual ranks observed
                            )

        # Add median annotations below the violins
        for i, median in enumerate(measure_medians):
            ax.text(i, 7, f'{median:.2f}',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontweight='bold',
                    fontsize=15,
                    color='#404040')  # Dark grey color

        # Customize plot
        axes[idx].set_title(criterion)
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel(DistanceMeasureCols.rank)

        # Rotate x-axis labels if they're too long
        axes[idx].tick_params(axis='x', rotation=45)

    # Remove any empty subplots if number of criteria < 6
    for idx in range(n_criteria, len(axes)):
        fig.delaxes(axes[idx])

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()
    return fig
