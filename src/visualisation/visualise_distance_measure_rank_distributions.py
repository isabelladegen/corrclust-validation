import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.evaluation.distance_metric_assessment import DistanceMeasureCols
from src.utils.distance_measures import short_distance_measure_names
from src.utils.plots.matplotlib_helper_functions import reset_matplotlib, Backends, fontsize


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
    if title is not "":
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
