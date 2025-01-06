import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats

from src.evaluation.describe_multiple_datasets import DescribeMultipleDatasets, DistParams
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends, reset_matplotlib


def get_distribution_bounds(dist_info: {}, data: np.ndarray, confidence_bounds: () = (0.001, 0.999)):
    """
    Calculate bounds for plotting considering both theoretical and empirical data.
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


def plot_standard_distributions(datasets: {}, dist_params: {}, fontsize=20, figsize: () = (20, 15),
                                backend: str = Backends.none.value):
    """
    Create a grid of distribution plots with QQ plot insets.

    Parameters:
    datasets: Dict where keys are variations ('RAW', 'NN', 'NC', 'RS') and values are lists of 2D numpy arrays
             Each array has shape (n_observations, n_timeseries)
    dist_params: Dict of distribution parameters for each time series
                 First level: time series name ('iob', 'cob', 'ig')
                 Second level: distribution parameters
    fontsize: int, font size for labels and titles
    figsize: tuple, figure size (width, height)
    """
    # Get dimensions and keys
    variations = list(datasets.keys())  # columns
    n_variations = len(variations)

    # Get time series names
    ts_names = list(dist_params.keys())
    n_timeseries = len(ts_names)

    # Setup plt
    reset_matplotlib(backend)

    # Create figure and grid
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_timeseries, n_variations, figure=fig)
    fig.suptitle('Distribution Fits Across Dataset Variations', fontsize=fontsize + 4)

    # Store x axis limits for consistency
    xlims = {ts_name: {'min': float('inf'), 'max': float('-inf')}
             for ts_name in ts_names}

    # First pass to determine consistent x-axis limits
    for ts_idx, ts_name in enumerate(ts_names):
        for variation in variations:
            data = datasets[variation]
            # Get all values for this time series from all datasets
            all_values = np.concatenate([d[:, ts_idx] for d in data])
            bounds = get_distribution_bounds(dist_params[ts_name], all_values)
            xlims[ts_name]['min'] = min(xlims[ts_name]['min'], bounds[0])
            xlims[ts_name]['max'] = max(xlims[ts_name]['max'], bounds[1])

    # Store y-axis limits for consistency
    ylims = {ts_name: {'min': float('inf'), 'max': float('-inf')}
             for ts_name in ts_names}

    # First pass to determine consistent y-axis limits
    for ts_idx, ts_name in enumerate(ts_names):
        # Use all values from the NN variation
        nn_data = datasets['NN']
        nn_all_values = np.concatenate([d[:, ts_idx] for d in nn_data])

        # Determine if distribution is continuous or discrete
        dist_method = dist_params[ts_name][DistParams.method]

        if dist_method.name in stats._continuous_distns._distn_names:
            # For continuous distributions, calculate PDF bounds
            x = np.linspace(*get_distribution_bounds(dist_params[ts_name], nn_all_values), 1000)

            # Calculate PDF at different parameter sets
            pdf_median = dist_method.pdf(
                x,
                *dist_params[ts_name][DistParams.median_args],
                **dist_params[ts_name][DistParams.median_kwargs]
            )
            pdf_min = dist_method.pdf(
                x,
                *dist_params[ts_name][DistParams.min_args],
                **dist_params[ts_name][DistParams.min_kwargs]
            )
            pdf_max = dist_method.pdf(
                x,
                *dist_params[ts_name][DistParams.max_args],
                **dist_params[ts_name][DistParams.max_kwargs]
            )

            ylims[ts_name] = {
                'min': 0,  # Density always starts at 0
                'max': max(pdf_median.max(), pdf_min.max(), pdf_max.max())
            }

        elif dist_method.name in stats._discrete_distns._distn_names:
            # For discrete distributions
            discrete_values = np.unique(np.floor(nn_all_values).astype(int))

            # Calculate PMF at different parameter sets
            pmf_median = dist_method.pmf(
                discrete_values,
                *dist_params[ts_name][DistParams.median_args],
                **dist_params[ts_name][DistParams.median_kwargs]
            )
            pmf_min = dist_method.pmf(
                discrete_values,
                *dist_params[ts_name][DistParams.min_args],
                **dist_params[ts_name][DistParams.min_kwargs]
            )
            pmf_max = dist_method.pmf(
                discrete_values,
                *dist_params[ts_name][DistParams.max_args],
                **dist_params[ts_name][DistParams.max_kwargs]
            )

            ylims[ts_name] = {
                'min': 0,  # Density always starts at 0
                'max': max(pmf_median.max(), pmf_min.max(), pmf_max.max())
            }

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
            all_values = np.concatenate([d[:, ts_idx] for d in data])

            # Plot histogram of actual values
            sns.histplot(all_values, stat='density', alpha=0.5, ax=ax, label='Empirical Distribution')

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
                ax.plot(x, pdf_median, 'r-', lw=2, label='Distribution (Median Parameters)')

                # Plot confidence bands
                pdf_min = dist_method.pdf(
                    x,
                    *dist_params[ts_name][DistParams.min_args],
                    **dist_params[ts_name][DistParams.min_kwargs]
                )
                pdf_max = dist_method.pdf(
                    x,
                    *dist_params[ts_name][DistParams.max_args],
                    **dist_params[ts_name][DistParams.max_kwargs]
                )
                ax.fill_between(x, pdf_min, pdf_max, alpha=0.2, color='r',
                                label='Distribution (Parameter Range)')

            elif dist_method.name in stats._discrete_distns._distn_names:
                # For discrete distributions
                discrete_values = np.unique(np.floor(all_values).astype(int))
                x = discrete_values

                # Plot median PMF
                pmf_median = dist_method.pmf(
                    x,
                    *dist_params[ts_name][DistParams.median_args],
                    **dist_params[ts_name][DistParams.median_kwargs]
                )
                ax.bar(x, pmf_median, alpha=0.5, color='r', label='Distribution (Median Parameters)')

                # Plot confidence bands for discrete distribution
                pmf_min = dist_method.pmf(
                    x,
                    *dist_params[ts_name][DistParams.min_args],
                    **dist_params[ts_name][DistParams.min_kwargs]
                )
                pmf_max = dist_method.pmf(
                    x,
                    *dist_params[ts_name][DistParams.max_args],
                    **dist_params[ts_name][DistParams.max_kwargs]
                )
                # Visualize confidence band for discrete distribution
                ax.fill_between(x, pmf_min, pmf_max, alpha=0.2, color='r',
                                label='Distribution (Parameter Range)')
            else:
                raise ValueError(f"Unsupported distribution type: {dist_method.name}")

            # Create inset for QQ plot
            axins = ax.inset_axes([0.65, 0.65, 0.3, 0.3])

            # Determine distribution type for QQ plot
            if dist_method.name in stats._continuous_distns._distn_names:
                # For continuous distributions
                p = np.linspace(0.01, 0.99, 100)
                theoretical_quantiles = dist_method.ppf(
                    p,
                    *dist_params[ts_name][DistParams.median_args],
                    **dist_params[ts_name][DistParams.median_kwargs]
                )
                empirical_quantiles = np.percentile(all_values, p * 100)

                axins.scatter(theoretical_quantiles, empirical_quantiles, alpha=0.5, s=10)

                # Add diagonal line and confidence bands to QQ plot
                qq_min = min(theoretical_quantiles.min(), empirical_quantiles.min())
                qq_max = max(theoretical_quantiles.max(), empirical_quantiles.max())
                axins.plot([qq_min, qq_max], [qq_min, qq_max], 'r--', alpha=0.8)

                # Add confidence bands to QQ plot
                theoretical_min = dist_method.ppf(
                    p,
                    *dist_params[ts_name][DistParams.min_args],
                    **dist_params[ts_name][DistParams.min_kwargs]
                )
                theoretical_max = dist_method.ppf(
                    p,
                    *dist_params[ts_name][DistParams.max_args],
                    **dist_params[ts_name][DistParams.max_kwargs]
                )
                axins.fill_between(theoretical_quantiles,
                                   theoretical_min, theoretical_max,
                                   alpha=0.1, color='r')

            elif dist_method.name in stats._discrete_distns._distn_names:
                # For discrete distributions
                unique_values = np.unique(np.floor(all_values).astype(int))
                probs = np.linspace(0.01, 0.99, len(unique_values))

                # Generate theoretical quantiles for discrete distribution
                theoretical_quantiles = dist_method.ppf(
                    probs,
                    *dist_params[ts_name][DistParams.median_args],
                    **dist_params[ts_name][DistParams.median_kwargs]
                )
                empirical_quantiles = np.percentile(all_values, probs * 100)

                axins.scatter(theoretical_quantiles, empirical_quantiles, alpha=0.5, s=10)

                # Add diagonal line
                qq_min = min(theoretical_quantiles.min(), empirical_quantiles.min())
                qq_max = max(theoretical_quantiles.max(), empirical_quantiles.max())
                axins.plot([qq_min, qq_max], [qq_min, qq_max], 'r--', alpha=0.8)

                # Add confidence bands to QQ plot for discrete distribution
                theoretical_min = dist_method.ppf(
                    probs,
                    *dist_params[ts_name][DistParams.min_args],
                    **dist_params[ts_name][DistParams.min_kwargs]
                )
                theoretical_max = dist_method.ppf(
                    probs,
                    *dist_params[ts_name][DistParams.max_args],
                    **dist_params[ts_name][DistParams.max_kwargs]
                )
                axins.fill_between(theoretical_quantiles,
                                   theoretical_min, theoretical_max,
                                   alpha=0.1, color='r')
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

            # Disable y-axis labels for subsequent columns in the same row
            if var_idx > 0:
                plt.setp(ax.get_yticklabels(), visible=False)

            ax.tick_params(labelsize=fontsize - 4)
            axins.tick_params(labelsize=fontsize - 8)

            # Add legend to the last subplot in the first row
            if ts_idx == n_timeseries - 1 and var_idx == n_variations - 1:
                # Place legend outside the plot on the right side
                ax.legend(fontsize=fontsize - 4, title='Distribution',
                          title_fontsize=fontsize - 4,
                          bbox_to_anchor=(1.05, 0), loc='lower left')

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

        fig = plot_standard_distributions(datasets=datasets, dist_params=dist_params, figsize=(20, 15),
                                          backend=self.backend)
        plt.show()
        return fig
