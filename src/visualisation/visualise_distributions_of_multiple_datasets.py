import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats

from src.evaluation.describe_multiple_datasets import DescribeMultipleDatasets, DistParams
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
        theoretical_min = dist_method.pdf(
            x,
            *dist_info[DistParams.min_args],
            **dist_info[DistParams.min_kwargs]
        )
        theoretical_max = dist_method.pdf(
            x,
            *dist_info[DistParams.max_args],
            **dist_info[DistParams.max_kwargs]
        )
    else:
        # For discrete distributions
        discrete_values = np.unique(np.floor(data).astype(int))
        # Calculate PMF at different parameter sets
        theoretical_min = dist_method.pmf(
            discrete_values,
            *dist_info[DistParams.min_args],
            **dist_info[DistParams.min_kwargs]
        )
        theoretical_max = dist_method.pmf(
            discrete_values,
            *dist_info[DistParams.max_args],
            **dist_info[DistParams.max_kwargs]
        )
    return 0, max(theoretical_min.max(), theoretical_max.max())  # pmf, pdf start at 0


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


def plot_standard_distributions(datasets: {}, dist_params: {}, reference_key: str = "NN", fontsize=20,
                                figsize: () = (20, 15), backend: str = Backends.none.value):
    """
    Create a grid of distribution plots with QQ plot insets.

    :param datasets: Dict where keys are variations ('RAW', 'NN', 'NC', 'RS') and values are lists of 2D numpy arrays
             Each array has shape (n_observations, n_timeseries)
    :param dist_params: Dict of distribution parameters for each time series
                 First level: time series name ('iob', 'cob', 'ig')
                 Second level: distribution parameters
    :param reference_key: key for data in datasets from which the dist_params are. Will be used to plot theoretical
    distribution and the bands in each square to visualise how far from the reference distribution the other
    variates are
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
    xlims = {ts_name: {'min': float('inf'), 'max': float('-inf')}
             for ts_name in ts_names}
    ylims = {ts_name: {'min': float('inf'), 'max': float('-inf')}
             for ts_name in ts_names}
    qq_lims = {ts_name: {'min': float('inf'), 'max': float('-inf')}
               for ts_name in ts_names}

    # First pass to determine limits
    for ts_idx, ts_name in enumerate(ts_names):
        reference_values_for_ts = all_data[reference_key][:, ts_idx]
        # Determine if distribution is continuous or discrete, throw error if neither
        dist_method = dist_params[ts_name][DistParams.method]

        # Calculate y-lims - this is only done on reference distribution values so some empirical values
        # might lay outside the y axis
        y_bounds = get_y_value_bounds(dist_params[ts_name], reference_values_for_ts, dist_method)
        ylims[ts_name]['min'] = y_bounds[0]
        ylims[ts_name]['max'] = y_bounds[1]

        for variation in variations:
            # Calculate x lims
            all_values_for_ts = all_data[variation][:, ts_idx]
            x_bounds = get_x_values_distribution_bounds(dist_params[ts_name], all_values_for_ts)
            xlims[ts_name]['min'] = min(xlims[ts_name]['min'], x_bounds[0])
            xlims[ts_name]['max'] = max(xlims[ts_name]['max'], x_bounds[1])

            # Calculate qq lims
            qq_bounds = get_qq_plots_bounds(dist_params[ts_name], all_values_for_ts, dist_method)
            qq_lims[ts_name]['min'] = min(qq_lims[ts_name]['min'], qq_bounds[0])
            qq_lims[ts_name]['max'] = max(qq_lims[ts_name]['max'], qq_bounds[1])

    # Create plots
    for ts_idx, ts_name in enumerate(ts_names):
        for var_idx, variation in enumerate(variations):
            # Create subplot with shared y-axis for the row
            if var_idx == 0:
                ax = fig.add_subplot(gs[ts_idx, var_idx])
                row_ax = ax  # Store the first axis for the row
            else:
                ax = fig.add_subplot(gs[ts_idx, var_idx], sharey=row_ax)

            # Get all values for this time series from current variation
            data = datasets[variation]
            all_values_for_ts = np.concatenate([d[:, ts_idx] for d in data])

            # Plot histogram of actual values
            sns.histplot(all_values_for_ts, stat='density', alpha=0.5, ax=ax, label='Empirical Distribution')

            # Set y-axis limits consistently
            ax.set_ylim(0, ylims[ts_name]['max'])

            # Determine if distribution is continuous or discrete
            dist_method = dist_params[ts_name][DistParams.method]

            # Prepare x values differently for continuous and discrete distributions
            if dist_method.name in stats._continuous_distns._distn_names:
                # For continuous distributions
                x = np.linspace(xlims[ts_name]['min'], xlims[ts_name]['max'], 1000)

                # Plot median distribution
                pdf_median = dist_method.pdf(
                    x,
                    *dist_params[ts_name][DistParams.median_args],
                    **dist_params[ts_name][DistParams.median_kwargs]
                )
                ax.plot(x, pdf_median, 'r-', lw=2, label='PDF/PMF Median NN Parameters')

            elif dist_method.name in stats._discrete_distns._distn_names:
                # For discrete distributions
                discrete_values = np.arange(
                    int(np.floor(xlims[ts_name]['min'])),
                    int(np.ceil(xlims[ts_name]['max'])) + 1
                )
                x = discrete_values

                # Plot median PMF
                pmf_median = dist_method.pmf(
                    x,
                    *dist_params[ts_name][DistParams.median_args],
                    **dist_params[ts_name][DistParams.median_kwargs]
                )
                ax.bar(x, pmf_median, alpha=0.5, color='r', label='PDF/PMF Median Parameters')
            else:
                raise ValueError(f"Unsupported distribution type: {dist_method.name}")

            # Create inset for QQ plot
            axins = ax.inset_axes([0.65, 0.6, 0.3, 0.3])

            # Determine distribution type for QQ plot
            if dist_method.name in stats._continuous_distns._distn_names:
                # For continuous distributions
                p = np.linspace(0.01, 0.99, 100)
                theoretical_quantiles = dist_method.ppf(
                    p,
                    *dist_params[ts_name][DistParams.median_args],
                    **dist_params[ts_name][DistParams.median_kwargs]
                )
                empirical_quantiles = np.percentile(all_values_for_ts, p * 100)

                # Plot the empirical vs theoretical points for this variation
                axins.scatter(theoretical_quantiles, empirical_quantiles, alpha=0.5, s=10)

                # Add diagonal line using the common limits for this row
                axins.plot([qq_lims[ts_name]['min'], qq_lims[ts_name]['max']],
                           [qq_lims[ts_name]['min'], qq_lims[ts_name]['max']],
                           'r--', alpha=0.8)

                # Calculate confidence bands based on NN case parameters
                x_range = np.linspace(qq_lims[ts_name]['min'], qq_lims[ts_name]['max'], 100)
                theoretical_quantiles_nn = x_range  # This is our reference line
                theoretical_min = dist_method.ppf(
                    dist_method.cdf(x_range, *dist_params[ts_name][DistParams.median_args]),
                    *dist_params[ts_name][DistParams.min_args],
                    **dist_params[ts_name][DistParams.min_kwargs]
                )
                theoretical_max = dist_method.ppf(
                    dist_method.cdf(x_range, *dist_params[ts_name][DistParams.median_args]),
                    *dist_params[ts_name][DistParams.max_args],
                    **dist_params[ts_name][DistParams.max_kwargs]
                )
                axins.fill_between(x_range, theoretical_min, theoretical_max,
                                   alpha=0.1, color='r')

                axins.set_xlim(qq_lims[ts_name]['min'], qq_lims[ts_name]['max'])
                axins.set_ylim(qq_lims[ts_name]['min'], qq_lims[ts_name]['max'])

            elif dist_method.name in stats._discrete_distns._distn_names:
                # For discrete distributions
                unique_values = np.unique(np.floor(all_values_for_ts).astype(int))
                probs = np.linspace(0.01, 0.99, len(unique_values))
                theoretical_quantiles = dist_method.ppf(
                    probs,
                    *dist_params[ts_name][DistParams.median_args],
                    **dist_params[ts_name][DistParams.median_kwargs]
                )
                empirical_quantiles = np.percentile(all_values_for_ts, probs * 100)

                # Plot the empirical vs theoretical points for this variation
                axins.scatter(theoretical_quantiles, empirical_quantiles, alpha=0.5, s=10)

                # Add diagonal line using the common limits for this row
                axins.plot([qq_lims[ts_name]['min'], qq_lims[ts_name]['max']],
                           [qq_lims[ts_name]['min'], qq_lims[ts_name]['max']],
                           'r--', alpha=0.8)

                # Calculate confidence bands based on NN case parameters
                x_range = np.arange(
                    int(np.floor(qq_lims[ts_name]['min'])),
                    int(np.ceil(qq_lims[ts_name]['max'])) + 1
                )
                theoretical_quantiles_nn = x_range  # This is our reference line
                theoretical_min = dist_method.ppf(
                    dist_method.cdf(x_range, *dist_params[ts_name][DistParams.median_args]),
                    *dist_params[ts_name][DistParams.min_args],
                    **dist_params[ts_name][DistParams.min_kwargs]
                )
                theoretical_max = dist_method.ppf(
                    dist_method.cdf(x_range, *dist_params[ts_name][DistParams.median_args]),
                    *dist_params[ts_name][DistParams.max_args],
                    **dist_params[ts_name][DistParams.max_kwargs]
                )
                axins.fill_between(x_range, theoretical_min, theoretical_max,
                                   alpha=0.1, color='r')

                axins.set_xlim(qq_lims[ts_name]['min'], qq_lims[ts_name]['max'])
                axins.set_ylim(qq_lims[ts_name]['min'], qq_lims[ts_name]['max'])
            else:
                raise ValueError(f"Unsupported distribution type: {dist_method.name}")

            axins.set_title('Q-Q Plot', fontsize=fontsize - 8)

            # Set consistent x-axis limits
            ax.set_xlim(xlims[ts_name]['min'], xlims[ts_name]['max'])

            # Labels and titles
            if ts_idx == 0:
                ax.set_title(variation, fontsize=fontsize)
            if var_idx == 0:
                ax.set_ylabel(f'{ts_name}\nDensity', fontsize=fontsize)
            else:
                # For all other columns, remove the label completely
                ax.set_ylabel('')

            # Disable y-axis labels for subsequent columns in the same row
            if var_idx > 0:
                plt.setp(ax.get_yticklabels(), visible=False)

            ax.tick_params(labelsize=fontsize - 4)
            axins.tick_params(labelsize=fontsize - 8)

            # Add legend to the last subplot in the first row
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
            column_name = SyntheticDataType.get_log_key_for_data_type(ds_type)
            ds = DescribeMultipleDatasets(wandb_run_file=run_file, overall_ds_name=overall_ds_name,
                                          data_type=ds_type, data_dir=self.data_dir, load_data=True)
            self.dataset_variates[column_name] = ds
        # dataset variate names
        self.col_names = [SyntheticDataType.get_log_key_for_data_type(ds_type) for ds_type in dataset_types]

    def plot_as_standard_distribution(self):
        datasets = {key: value.get_list_of_xtrain_of_all_datasets() for key, value in self.dataset_variates.items()}
        nn_col_name = SyntheticDataType.get_log_key_for_data_type(SyntheticDataType.non_normal_correlated)
        if nn_col_name in self.dataset_variates.keys():
            dist_params = self.dataset_variates[nn_col_name].get_median_min_max_distribution_parameters()
        else:
            dist_params = list(self.dataset_variates.values())[0].get_median_min_max_distribution_parameters()

        fig = plot_standard_distributions(datasets=datasets, dist_params=dist_params, figsize=(20, 12),
                                          backend=self.backend)
        plt.show()
        return fig
