from os import path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from scipy import stats

from src.evaluation.describe_subjects_for_data_variant import DescribeSubjectsForDataVariant, DistParams
from src.utils.configurations import base_dataset_result_folder_for_type, ResultsType, get_image_name_based_on_data_dir, \
    OVERALL_DISTRIBUTION_IMAGE
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends, reset_matplotlib


def get_x_values_distribution_bounds(dist_info: {}, data: np.ndarray, confidence_bounds: () = (0.001, 0.999)):
    """
    Calculate x value bounds for plotting considering both theoretical and empirical data and return min and max.
    """
    lower_theoretical = dist_info[DistParams.method].ppf(
        confidence_bounds[0],
        *dist_info[DistParams.min_args],
        **dist_info[DistParams.min_kwargs]
    )
    upper_theoretical = dist_info[DistParams.method].ppf(
        confidence_bounds[1],
        *dist_info[DistParams.max_args],
        **dist_info[DistParams.max_kwargs]
    )

    lower_empirical = np.percentile(data, confidence_bounds[0] * 100)
    upper_empirical = np.percentile(data, confidence_bounds[1] * 100)

    return (min(lower_theoretical, lower_empirical),
            max(upper_theoretical, upper_empirical))


def get_y_value_bounds(dist_info: {}, data: np.ndarray, dist_method):
    """ Bounds for y values for plotting, only considering theoretical distribution"""
    is_continuous = dist_method.name in stats._continuous_distns._distn_names
    if not is_continuous:  # should be discrete otherwise throw error
        error_msg = "Unsupported distribution with name: " + str(dist_method)
        assert dist_method.name in stats._discrete_distns._distn_names, error_msg

    if is_continuous:
        # For continuous distributions, calculate PDF bounds
        x = np.linspace(*get_x_values_distribution_bounds(dist_info, data), 1000)
        median_args = dist_info[DistParams.median_args]
        median_kwargs = dist_info[DistParams.median_kwargs]
        pdf_vals = dist_method.pdf(x, *median_args, **median_kwargs)
        theoretical_max = pdf_vals.max() * 1.1
    else:
        # For discrete distributions
        x = np.arange(0, int(data.max()) + 1)
        median_args = dist_info[DistParams.median_args]
        median_kwargs = dist_info[DistParams.median_kwargs]
        pmf_vals = dist_method.pmf(x, *median_args, **median_kwargs)
        theoretical_max = pmf_vals.max() * 1.2
    hist_vals, _ = np.histogram(data, bins=100, density=True)
    empirical_max = hist_vals.max() * 1.2  # Add 20% padding
    return 0, max(empirical_max, theoretical_max)  # pmf, pdf start at 0


def get_qq_plots_bounds(dist_info: {}, data: np.ndarray, dist_method):
    """
    Calculate y value bounds for plotting considering both theoretical and empirical data and return min and max.
    Using either ppf or pmf to estimate bound
    """
    is_continuous = dist_method.name in stats._continuous_distns._distn_names
    if not is_continuous:  # should be discrete otherwise throw error
        error_msg = "Unsupported distribution with name: " + str(dist_method)
        assert dist_method.name in stats._discrete_distns._distn_names, error_msg

    if is_continuous:
        p = np.linspace(0.01, 0.99, 100)
        theoretical_quantiles = dist_method.ppf(
            p,
            *dist_info[DistParams.median_args],
            **dist_info[DistParams.median_kwargs]
        )
        empirical_quantiles = np.percentile(data, p * 100)
    else:  # discrete case
        unique_values = np.unique(np.floor(data).astype(int))
        probs = np.linspace(0.01, 0.99, len(unique_values))
        theoretical_quantiles = dist_method.ppf(
            probs,
            *dist_info[DistParams.median_args],
            **dist_info[DistParams.median_kwargs]
        )
        empirical_quantiles = np.percentile(data, probs * 100)
    return (min(theoretical_quantiles.min(), empirical_quantiles.min()),
            max(theoretical_quantiles.max(), empirical_quantiles.max()))


def plot_standard_distributions(datasets: {}, dist_params: {}, reference_keys: str = ["Non-normal", "Downsampled"],
                                show_legend: bool = False, fontsize=20,
                                figsize: () = (24, 14), backend: str = Backends.none.value):
    """
    Create a grid of distribution plots with QQ plot insets.

    :param datasets: Dict where keys are variations ('Raw', 'Correlated', 'Non-normal', 'Downsampled') and values are lists of 2D numpy arrays
             Each array has shape (n_observations, n_timeseries)
    :param dist_params: Dict of distribution parameters for each time series
                 First level: time series name ('iob', 'cob', 'ig')
                 Second level: distribution parameters
    :param reference_key: key for data in datasets from which the dist_params are. Will be used to plot theoretical
    distribution for non-normal variant, other variants are shown for standard normal distribution
    :param show_legend: whether to show legend box
    :param fontsize: int, font size for labels and titles
    :param figsize: tuple, figure size (width, height)
    """
    # Get dimensions and keys - columns
    variations = list(datasets.keys())
    n_variations = len(variations)

    # Get time series names - rows
    ts_names = list(dist_params.keys())
    n_timeseries = len(ts_names)

    # Setup plt
    reset_matplotlib(backend)

    # Create figure and grid
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_timeseries, n_variations, figure=fig)

    # Concatenate data values for each dataset and each variation
    all_data = {}  # keys variation, values 2d concatenated np.array
    for variation in variations:
        # concatenate all values from all datasets in this variation
        all_data[variation] = np.concatenate([d for d in datasets[variation]])

    # Calculate various axes limits - allows comparing columns
    # Keep x and y axes  and qq plot size and axes consistent for each ts variate (rows)
    # Default limits
    # Calculate various axes limits
    # We'll handle each variation type differently
    xlims = {ts_name: {variation: {'min': 0, 'max': 0} for variation in variations}
             for ts_name in ts_names}
    ylims = {ts_name: {variation: {'min': 0, 'max': 0} for variation in variations}
             for ts_name in ts_names}
    qq_lims = {ts_name: {variation: {'min': 0, 'max': 0} for variation in variations}
               for ts_name in ts_names}

    # First pass to determine limits
    for ts_idx, ts_name in enumerate(ts_names):
        for variation in variations:
            all_values_for_ts = all_data[variation][:, ts_idx]

            if variation in reference_keys:
                # For Non-normal and for the complete downsampled, use the specific distribution parameters
                dist_method = dist_params[ts_name][DistParams.method]
                dist_info = dist_params[ts_name]

                # Calculate bounds based on the specific distribution
                x_bounds = get_x_values_distribution_bounds(dist_info, all_values_for_ts)
                y_bounds = get_y_value_bounds(dist_info, all_values_for_ts, dist_method)
                qq_bounds = get_qq_plots_bounds(dist_info, all_values_for_ts, dist_method)
            else:
                # For Raw, Correlated, and partial/sparse Downsampled, use standard normal with adjustments
                # Calculate empirical max density for y-axis
                hist_vals, _ = np.histogram(all_values_for_ts, bins=50, density=True)
                max_density = max(hist_vals.max(), 0.4)
                x_bounds = (-3, 3)  # Covers 99.7% of standard normal
                y_bounds = (0, max_density * 1.1)  # Add 10% padding above max density
                qq_bounds = (-3, 3)  # Standard normal quantiles

            xlims[ts_name][variation]['min'] = x_bounds[0]
            xlims[ts_name][variation]['max'] = x_bounds[1]
            ylims[ts_name][variation]['min'] = y_bounds[0]
            ylims[ts_name][variation]['max'] = y_bounds[1]
            qq_lims[ts_name][variation]['min'] = qq_bounds[0]
            qq_lims[ts_name][variation]['max'] = qq_bounds[1]

    # Create plots
    for ts_idx, ts_name in enumerate(ts_names):
        for var_idx, variation in enumerate(variations):
            ax = fig.add_subplot(gs[ts_idx, var_idx])

            # Set x and y axes limits consistently
            ax.set_ylim(ylims[ts_name][variation]['min'], ylims[ts_name][variation]['max'])
            ax.set_xlim(xlims[ts_name][variation]['min'], xlims[ts_name][variation]['max'])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

            # Get all values for this time series from current variation
            all_values_for_ts = all_data[variation][:, ts_idx]

            # Plot density histogram of empirical values
            sns.histplot(all_values_for_ts, stat='density', alpha=0.5, ax=ax, label='Empirical')

            if variation in reference_keys:
                # Use the non-normal distribution parameters
                dist_method = dist_params[ts_name][DistParams.method]
                median_args = dist_params[ts_name][DistParams.median_args]
                median_kwargs = dist_params[ts_name][DistParams.median_kwargs]
            else:
                # Use standard normal distribution (μ=0, σ=1)
                dist_method = stats.norm
                median_args = (0, 1)
                median_kwargs = {}

            # Plot theoretical distribution for median parameters
            is_continuous = dist_method.name in stats._continuous_distns._distn_names
            if not is_continuous:  # should be discrete otherwise throw error
                error_msg = "Unsupported distribution with name: " + str(dist_method)
                assert dist_method.name in stats._discrete_distns._distn_names, error_msg

            if is_continuous:
                # For continuous distributions
                x = np.linspace(xlims[ts_name][variation]['min'], xlims[ts_name][variation]['max'], 1000)
                pdf_median = dist_method.pdf(x, *median_args, **median_kwargs)
                ax.plot(x, pdf_median, 'r-', lw=2, label='PDF/PMF Median Target Distribution')
            else:
                # For discrete distributions
                x = np.arange(int(np.floor(xlims[ts_name][variation]['min'])),
                              int(np.ceil(xlims[ts_name][variation]['max'])) + 1)
                pmf_median = dist_method.pmf(x, *median_args, **median_kwargs)
                ax.bar(x, pmf_median, alpha=0.5, color='r', label='PDF/PMF Median Target Distribution')

            # Create inset for QQ plot with limits
            axins = ax.inset_axes([0.55, 0.51, 0.4, 0.4])  # x, y, width, height
            axins.set_xlim(qq_lims[ts_name][variation]['min'], qq_lims[ts_name][variation]['max'])
            axins.set_ylim(qq_lims[ts_name][variation]['min'], qq_lims[ts_name][variation]['max'])

            # Add diagonal line using the common limits for this row - theoretical distribution
            axins.plot([qq_lims[ts_name][variation]['min'], qq_lims[ts_name][variation]['max']],
                       [qq_lims[ts_name][variation]['min'], qq_lims[ts_name][variation]['max']],
                       'r--', alpha=0.8)

            # Set title
            axins.set_title('Q-Q Plot', fontsize=fontsize - 8)

            # Calculate probs for continuous or discrete case
            if is_continuous:
                probs = np.linspace(0.001, 0.999, 100)
            else:
                unique_values = np.unique(np.floor(all_values_for_ts).astype(int))
                probs = np.linspace(0.001, 0.999, len(unique_values))

            # Calculate empirical quantiles
            empirical_quantiles = np.percentile(all_values_for_ts, probs * 100)

            # Calculate main diagonal (median parameters) as before
            if variation in reference_keys:
                theoretical_quantiles = dist_method.ppf(probs, *median_args, **median_kwargs)
            else:
                # Use standard normal distribution for theoretical quantiles
                theoretical_quantiles = stats.norm.ppf(probs, 0, 1)

            # Plot QQ points
            axins.scatter(theoretical_quantiles, empirical_quantiles, alpha=0.5, s=10)

            # Set dataset variation as title
            if ts_idx == 0:
                ax.set_title(variation, fontsize=fontsize, fontweight='bold')

            # Set y labels for first column only
            if var_idx == 0:
                ax.set_ylabel(ts_name.upper(), fontsize=fontsize, fontweight='bold')
            else:
                ax.set_ylabel('')

            ax.tick_params(labelsize=fontsize - 4)
            axins.tick_params(labelsize=fontsize - 8)

            # Add legend to the last subplot in the first row
            if show_legend:
                if ts_idx == n_timeseries - 1 and var_idx == n_variations - 1:
                    # Place legend outside the plot on the right side
                    ax.legend(fontsize=fontsize - 4, title_fontsize=fontsize - 4, bbox_to_anchor=(1.05, 0),
                              loc='lower left')

    plt.tight_layout()
    return fig


class VisualiseDistributionsOfMultipleDatasets:
    """Use this class for visualising data properties such as distributions from multiple datasets"""

    def __init__(self, run_file: str, overall_ds_name: str, dataset_types: [str], data_dir: str,
                 backend: str = Backends.none.value):
        self.run_file = run_file
        self.overall_ds_name = overall_ds_name
        self.dataset_types = dataset_types
        self.data_dir = data_dir
        self.backend = backend
        self.row_names = []
        self.dataset_variates = {}
        for ds_type in dataset_types:
            column_name = SyntheticDataType.get_display_name_for_data_type(ds_type)
            ds = DescribeSubjectsForDataVariant(wandb_run_file=run_file, overall_ds_name=overall_ds_name,
                                                data_type=ds_type, data_dir=self.data_dir, load_data=True)
            self.dataset_variates[column_name] = ds
        # dataset variate names
        self.col_names = [SyntheticDataType.get_display_name_for_data_type(ds_type) for ds_type in dataset_types]

    def plot_as_standard_distribution(self, root_result_dir: str, use_nn_params_for: [] = ["Non-normal", "Downsampled"],
                                      save_fig: bool = False):
        """
            Creates a plot of the dataset variation as columns (Raw, Correlated, Non-normal, Downsampled) and the time series as rows (IOB, COB,
            IG). Each square show the theoretical PDF/PMF distribution of the median parameters for the distribution
            we shifted to in NN, the empirical observations in density space and a QQ-plot inset. We can see
            what the ds variations do to the specified distribution for NN.
            :param root_result_dir: root result dir to save the figure this will be put in the dataset-description
            :param use_nn_params_for: generation stages for which to plot the non-normal distributions
            :param save_fig: whether to save the figure
            :return: fig
        """
        datasets = {key: value.get_list_of_xtrain_of_all_datasets() for key, value in self.dataset_variates.items()}
        nn_col_name = SyntheticDataType.get_display_name_for_data_type(SyntheticDataType.non_normal_correlated)
        if nn_col_name in self.dataset_variates.keys():
            dist_params = self.dataset_variates[nn_col_name].get_median_min_max_distribution_parameters()
        else:
            dist_params = list(self.dataset_variates.values())[0].get_median_min_max_distribution_parameters()

        fig = plot_standard_distributions(datasets=datasets, dist_params=dist_params, reference_keys=use_nn_params_for,
                                          figsize=(24, 16),
                                          backend=self.backend)
        plt.show()
        if save_fig:
            folder = base_dataset_result_folder_for_type(root_result_dir, ResultsType.dataset_description)
            fig.savefig(path.join(folder, get_image_name_based_on_data_dir(OVERALL_DISTRIBUTION_IMAGE, self.data_dir)),
                        dpi=300, bbox_inches='tight')
        return fig
