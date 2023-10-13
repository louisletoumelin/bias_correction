"""ALE plotting for continuous or categorical features."""
from collections.abc import Iterable
from functools import reduce
from itertools import product
from operator import add
import uuid

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cbook
import numpy as np
from numpy import ma
import pandas as pd

try:
    import seaborn as sns

    _sns = True
except ModuleNotFoundError:
    _sns = False
from matplotlib.patches import Rectangle
import scipy
from scipy.spatial import cKDTree

from bias_correction.train.utils import create_folder_if_doesnt_exist
from bias_correction.train.metrics import bias_direction

__all__ = ("ale_plot",)


class MidPointNorm(Normalize):
    def __init__(self, midpoint=0, vmin=None, vmax=None, clip=False):
        Normalize.__init__(self,vmin, vmax, clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if not (vmin < midpoint < vmax):
            raise ValueError("midpoint must be between maxvalue and minvalue.")
        elif vmin == vmax:
            result.fill(0) # Or should it be all masked? Or 0.5?
        elif vmin > vmax:
            raise ValueError("maxvalue must be bigger than minvalue")
        else:
            vmin = float(vmin)
            vmax = float(vmax)
            if clip:
                mask = ma.getmask(result)
                result = ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                  mask=mask)

            # ma division is very slow; we can take a shortcut
            resdat = result.data

            #First scale to -1 to 1 range, than to from 0 to 1.
            resdat -= midpoint
            resdat[resdat>0] /= abs(vmax - midpoint)
            resdat[resdat<0] /= abs(vmin - midpoint)

            resdat /= 2.
            resdat += 0.5
            result = ma.array(resdat, mask=result.mask, copy=False)

        if is_scalar:
            result = result[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if cbook.iterable(value):
            val = ma.asarray(value)
            val = 2 * (val-0.5)
            val[val>0]  *= abs(vmax - midpoint)
            val[val<0] *= abs(vmin - midpoint)
            val += midpoint
            return val
        else:
            val = 2 * (value - 0.5)
            if val < 0:
                return  val*abs(vmin-midpoint) + midpoint
            else:
                return  val*abs(vmax-midpoint) + midpoint



def _parse_features(features):
    """Standardise representation of column labels.
    Args:
        features : object
            One or more column labels.
    Returns:
        features : array-like
            An array of input features.
    Examples
    --------
    >>> _parse_features(1)
    array([1])
    >>> _parse_features(('a', 'b'))
    array(['a', 'b'], dtype='<U1')
    """
    if isinstance(features, Iterable) and not isinstance(features, str):
        # If `features` is a non-string iterable.
        return np.asarray(features)
    else:
        # If `features` is not an iterable, or it is a string, then assume it
        # represents one column label.
        return np.asarray([features])


def _check_two_ints(values):
    """Retrieve two integers.
    Parameters
    ----------
    values : [2-iterable of] int
        Values to process.
    Returns
    -------
    values : 2-tuple of int
        The processed integers.
    Raises
    ------
    ValueError
        If more than 2 values are given.
    ValueError
        If the values are not integers.
    Examples
    --------
    >>> _check_two_ints(1)
    (1, 1)
    >>> _check_two_ints((1, 2))
    (1, 2)
    >>> _check_two_ints((1,))
    (1, 1)
    """
    if isinstance(values, (int, np.integer)):
        values = (values, values)
    elif len(values) == 1:
        values = (values[0], values[0])
    elif len(values) != 2:
        raise ValueError(
            "'{}' values were given. Expected at most 2.".format(len(values))
        )

    if not all(isinstance(n_bin, (int, np.integer)) for n_bin in values):
        raise ValueError(
            "All values must be an integer. Got types '{}' instead.".format(
                {type(n_bin) for n_bin in values}
            )
        )
    return values


def _get_centres(x):
    """Return bin centres from bin edges.
    Parameters
    ----------
    x : array-like
        The first axis of `x` will be averaged.
    Returns
    -------
    centres : array-like
        The centres of `x`, the shape of which is (N - 1, ...) for
        `x` with shape (N, ...).
    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0, 1, 2, 3])
    >>> _get_centres(x)
    array([0.5, 1.5, 2.5])
    """
    return (x[1:] + x[:-1]) / 2


def _ax_title(ax, title, subtitle=""):
    """Add title to axis.
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to add title to.
    title : str
        Axis title.
    subtitle : str, optional
        Sub-title for figure. Will appear one line below `title`.
    """
    ax.set_title("\n".join((title, subtitle)))


def _ax_labels(ax, xlabel=None, ylabel=None):
    """Add labels to axis.
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to add labels to.
    xlabel : str, optional
        X axis label.
    ylabel : str, optional
        Y axis label.
    """
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)


def _ax_quantiles(ax, quantiles, twin="x"):
    """Plot quantiles of a feature onto axis.
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to modify.
    quantiles : array-like
        Quantiles to plot.
    twin : {'x', 'y'}, optional
        Select the axis for which to plot quantiles.
    Raises
    ------
    ValueError
        If `twin` is not one of 'x' or 'y'.
    """
    if twin not in ("x", "y"):
        raise ValueError("'twin' should be one of 'x' or 'y'.")

    # Duplicate the 'opposite' axis so we can define a distinct set of ticks for the
    # desired axis (`twin`).
    ax_mod = ax.twiny() if twin == "x" else ax.twinx()

    # Set the new axis' ticks for the desired axis.
    getattr(ax_mod, "set_{twin}ticks".format(twin=twin))(quantiles)
    # Set the corresponding tick labels.

    # Calculate tick label percentage values for each quantile (bin edge).
    percentages = (
        100 * np.arange(len(quantiles), dtype=np.float64) / (len(quantiles) - 1)
    )

    # If there is a fractional part, add a decimal place to show (part of) it.
    fractional = (~np.isclose(percentages % 1, 0)).astype("int8")

    getattr(ax_mod, "set_{twin}ticklabels".format(twin=twin))(
        [
            "{0:0.{1}f}%".format(percent, format_fraction)
            for percent, format_fraction in zip(percentages, fractional)
        ],
        color="#545454",
        fontsize=7,
    )
    getattr(ax_mod, "set_{twin}lim".format(twin=twin))(
        getattr(ax, "get_{twin}lim".format(twin=twin))()
    )


def _first_order_quant_plot(ax, quantiles, ale, exp=None, feature="", folder_name="", ale_std=None, ale_std_upper=None, ale_std_lower=None, color="C0", **kwargs):
    """First order ALE plot.
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot onto.
    quantiles : array-like
        ALE quantiles.
    ale : array-like
        ALE to plot.
    **kwargs : plot properties, optional
        Additional keyword parameters are passed to `ax.plot`.
    """
    if (ale_std_upper is not None) and (np.array(ale_std_upper).size > 0):
        print("debug _first_order_quant_plot")
        print("ale_std_upper")
        print(ale_std_upper)
        print("ale")
        print(ale)
        ale_std_upper = ale_std_upper.astype(np.float32)
        ale_std_lower = ale_std_lower.astype(np.float32)
        ale = ale.astype(np.float32)
        ax.fill_between(_get_centres(quantiles), ale_std_lower, ale_std_upper, alpha=0.2, color=color)

    ax.plot(_get_centres(quantiles), ale, color=color, **kwargs)


def _second_order_quant_plot(
    fig, ax, quantiles_list, ale, mark_empty=True, n_interp=50, **kwargs
):
    """Second order ALE plot.
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot onto.
    quantiles_list : array-like
        ALE quantiles for the first (`quantiles_list[0]`) and second
        (`quantiles_list[1]`) features.
    ale : masked array
        ALE to plot. Where `ale.mask` is True, this denotes bins where no samples were
        available. See `mark_empty`.
    mark_empty : bool, optional
        If True, plot rectangles over bins that did not contain any samples.
    n_interp : [2-iterable of] int, optional
        The number of interpolated samples generated from `ale` prior to contour
        plotting. Two integers may be given to specify different interpolation steps
        for the two features.
    **kwargs : contourf properties, optional
        Additional keyword parameters are passed to `ax.contourf`.
    Raises
    ------
    ValueError
        If `n_interp` values were not integers.
    ValueError
        If more than 2 values were given for `n_interp`.
    """
    centres_list = [_get_centres(quantiles) for quantiles in quantiles_list]
    n_x, n_y = _check_two_ints(n_interp)
    x = np.linspace(centres_list[0][0], centres_list[0][-1], n_x)
    y = np.linspace(centres_list[1][0], centres_list[1][-1], n_y)

    X, Y = np.meshgrid(x, y, indexing="xy")
    ale_interp = scipy.interpolate.interp2d(centres_list[0], centres_list[1], ale.T)
    CF = ax.contourf(X, Y, ale_interp(x, y), cmap="bwr", norm=MidPointNorm(midpoint=0), levels=30, alpha=0.7, **kwargs)

    if mark_empty and np.any(ale.mask):
        # Do not autoscale, so that boxes at the edges (contourf only plots the bin
        # centres, not their edges) don't enlarge the plot.
        plt.autoscale(False)
        # Add rectangles to indicate cells without samples.
        for i, j in zip(*np.where(ale.mask)):
            ax.add_patch(
                Rectangle(
                    [quantiles_list[0][i], quantiles_list[1][j]],
                    quantiles_list[0][i + 1] - quantiles_list[0][i],
                    quantiles_list[1][j + 1] - quantiles_list[1][j],
                    linewidth=0.5,
                    edgecolor="k",
                    facecolor="black",
                    alpha=1,
                )
            )
    fig.colorbar(CF)


def _get_quantiles(train_set, feature, bins):
    """Get quantiles from a feature in a dataset.
    Parameters
    ----------
    train_set : pandas.core.frame.DataFrame
        Dataset containing feature `feature`.
    feature : column label
        Feature for which to calculate quantiles.
    bins : int
        The number of quantiles is calculated as `bins + 1`.
    Returns
    -------
    quantiles : array-like
        Quantiles.
    bins : int
        Number of bins, `len(quantiles) - 1`. This may be lower than the original
        `bins` if identical quantiles were present.
    Raises
    ------
    ValueError
        If `bins` is not an integer.
    Notes
    -----
    When using this definition of quantiles in combination with a half open interval
    (lower quantile, upper quantile], care has to taken that the smallest observation
    is included in the first bin. This is handled transparently by `np.digitize`.
    """
    if not isinstance(bins, (int, np.integer)):
        raise ValueError(
            "Expected integer 'bins', but got type '{}'.".format(type(bins))
        )
    quantiles = np.unique(
        np.quantile(
            train_set[feature], np.linspace(0, 1, bins + 1), interpolation="lower"
        )
    )
    bins = len(quantiles) - 1
    return quantiles, bins


def _first_order_ale_quant(predictor, train_set, feature, bins,
                           use_std=True,
                           type_of_output="speed",
                           only_local_effects=False,
                           data_loader=None):
    """Estimate the first-order ALE function for single continuous feature data.
    Parameters
    ----------
    predictor : callable
        Prediction function.
    train_set : pandas.core.frame.DataFrame
        Training set on which the model was trained.
    feature : column label
        Feature name. A single column label.
    bins : int
        This defines the number of bins to compute. The effective number of bins may
        be less than this as only unique quantile values of train_set[feature] are
        used.
    Returns
    -------
    ale : array-like
        The first order ALE.
    quantiles : array-like
        The quantiles used.
    """
    quantiles, _ = _get_quantiles(train_set, feature, bins)

    # Define the bins the feature samples fall into. Shift and clip to ensure we are
    # getting the index of the left bin edge and the smallest sample retains its index
    # of 0.
    indices = np.clip(
        np.digitize(train_set[feature], quantiles, right=True) - 1, 0, None
    )

    # Assign the feature quantile values to two copied training datasets, one for each
    # bin edge. Then compute the difference between the corresponding predictions
    predictions = []
    for offset in range(2):
        mod_train_set = train_set.copy()
        mod_train_set[feature] = quantiles[indices + offset]
        if data_loader is not None:
            #tf_inputs = data_loader.get_tf_zipped_inputs(inputs=mod_train_set).batch(data_loader.length_test)
            inputs_test = data_loader.get_tf_zipped_inputs(mode="test", inputs=mod_train_set).batch(128)
            # predictor = predict_multiple_batches(inputs,
            #                                  model_version="last",
            #                                  batch_size=1,
            #                                  output_shape=(2, 360, 2761, 2761),
            #                                  force_build=False)
            predictions.append(predictor(inputs_test))
            #for i in tf_inputs:
            #    pred = predictor(i)
            #    # Intermediate outputs are not considered
            #    pred = np.squeeze(pred[0])
        #predictions.append(pred)

    # The individual effects.
    if type_of_output == "speed":
        effects = predictions[1] - predictions[0]
    elif type_of_output == "dir":
        effects = bias_direction(predictions[0], predictions[1])

    # Average these differences within each bin.
    index_groupby = pd.DataFrame({"index": indices, "effects": effects}).groupby("index")

    mean_effects = index_groupby.mean().to_numpy().flatten()

    """
    if use_std:
        mean_effects_std = index_groupby.std().to_numpy().flatten()
        if not only_local_effects:
            ale_std = np.array([0, *np.cumsum(mean_effects_std)])
        else:
            ale_std = np.array([0, mean_effects_std])
        ale_std = _get_centres(ale_std)
        if not only_local_effects:
            ale_std -= np.sum(ale_std * index_groupby.size() / train_set.shape[0])
    """

    if use_std:
        mean_effects_std = index_groupby.std().to_numpy().flatten()
        mean_effects_lower = mean_effects - mean_effects_std
        mean_effects_upper = mean_effects + mean_effects_std

        if not only_local_effects:
            ale_std = np.array([0, *np.cumsum(mean_effects_std)])  # updated
            ale_std_lower = np.array([0, *np.cumsum(mean_effects_lower)])
            ale_std_upper = np.array([0, *np.cumsum(mean_effects_upper)])
        else:
            ale_std = np.array([0, *mean_effects_std])

        if not only_local_effects:
            ale_std = _get_centres(ale_std)
            ale_std_lower = _get_centres(ale_std_lower)
            ale_std_upper = _get_centres(ale_std_upper)

    if not only_local_effects:
        ale = np.array([0, *np.cumsum(mean_effects)])
    else:
        #ale = np.array([0, *mean_effects])
        ale = mean_effects

    # The uncentred mean main effects at the bin centres.
    if not only_local_effects:
        ale = _get_centres(ale)

    # Centre the effects by subtracting the mean (the mean of the individual
    # `effects`, which is equivalently calculated using `mean_effects` and the number
    # of samples in each bin).

    if use_std:
        if not only_local_effects:
            ale_std -= np.sum(ale_std * index_groupby.size() / train_set.shape[0])
            ale_std_lower -= np.sum(ale * index_groupby.size() / train_set.shape[0])
            ale_std_upper -= np.sum(ale * index_groupby.size() / train_set.shape[0])

    if not only_local_effects:
        ale -= np.sum(ale * index_groupby.size() / train_set.shape[0])

    if use_std:
        if not only_local_effects:
            print("debug _first_order_ale_quant")
            print("ale_std")
            print(ale_std_lower)
            print(ale_std_upper)
            return ale, quantiles, ale_std_lower, ale_std_upper
        else:
            return ale, quantiles, None, None
    else:
        return ale, quantiles


def _second_order_ale_quant(predictor, train_set, features, bins, data_loader=None, type_of_output="speed"):
    """Estimate the second-order ALE function for two continuous feature data.

    Parameters
    ----------
    predictor : callable
        Prediction function.
    train_set : pandas.core.frame.DataFrame
        Training set on which the model was trained.
    features : 2-iterable of column label
        The two desired features, as two column labels.
    bins : [2-iterable of] int
        This defines the number of bins to compute. The effective number of bins may
        be less than this as only unique quantile values of train_set[feature] are
        used. If one integer is given, this is used for both features.
    Returns
    -------
    ale : (M, N) masked array
        The second order ALE. Elements are masked where no data was available.
    quantiles : 2-tuple of array-like
        The quantiles used: first the quantiles for `features[0]` with shape (M + 1,),
        then for `features[1]` with shape (N + 1,).
    Raises
    ------
    ValueError
        If `features` does not contain 2 features.
    ValueError
        If more than 2 bins are given.
    ValueError
        If bins are not integers.
    """
    features = _parse_features(features)
    if len(features) != 2:
        raise ValueError(
            "'features' contained '{n_feat}' features. Expected 2.".format(
                n_feat=len(features)
            )
        )

    """
    quantiles_list = (array([-2.48515187e-03, -1.18396047e-03, -1.02171965e-03, -6.49426540e-04,
        -3.99676093e-04, -2.79030995e-04, -2.64383736e-04, -1.45108002e-04,
        -7.36356815e-05, -6.21087092e-05,  0.00000000e+00,  1.08923545e-04,
         1.32365720e-04,  2.43507369e-04,  2.55341962e-04,  3.15633544e-04,
         4.02168109e-04,  1.30754535e-03,  2.13019550e-03,  5.92331029e-03,
         9.84235574e-03]), array([-0.08663303, -0.06101183, -0.01994195, -0.01271159, -0.00376492,
        -0.00344781, -0.00228665, -0.00163859, -0.00123915, -0.00101997,
         0.        ,  0.00023709,  0.00060737,  0.00153958,  0.00249078,
         0.00293294,  0.00438314,  0.00607964,  0.00956489,  0.01542969,
         0.02629123]))
    
    bins_list = (20, 20)
    """
    quantiles_list, bins_list = tuple(
        zip(
            *(
                _get_quantiles(train_set, feature, n_bin)
                for feature, n_bin in zip(features, _check_two_ints(bins))
            )
        )
    )

    # Define the bins the feature samples fall into. Shift and clip to ensure we are
    # getting the index of the left bin edge and the smallest sample retains its index
    # of 0.
    indices_list = [
        np.clip(np.digitize(train_set[feature], quantiles, right=True) - 1, 0, None)
        for feature, quantiles in zip(features, quantiles_list)
    ]
    """
    indices_list = [array([8, 6, 1, ..., 6, 2, 3]), array([11, 12, 17, ..., 12, 16, 15])]
    """

    # Invoke the predictor at the corners of the bins. Then compute the second order
    # difference between the predictions at the bin corners.
    predictions = {}
    for shifts in product(*(range(2),) * 2):  # shifts in ((0, 0), (0, 1), (1, 0), (1, 1))
        mod_train_set = train_set.copy()
        for i in range(2):
            mod_train_set[features[i]] = quantiles_list[i][indices_list[i] + shifts[i]]
        inputs_test = data_loader.get_tf_zipped_inputs(mode="test", inputs=mod_train_set).batch(128)
        predictions[shifts] = predictor(inputs_test)

    # The individual effects.
    if type_of_output == "speed":
        effects = (predictions[(1, 1)] - predictions[(1, 0)]) - (predictions[(0, 1)] - predictions[(0, 0)])
    else:
        effects = bias_direction(predictions[(1, 0)], predictions[(1, 1)]) - bias_direction(predictions[(0, 0)], predictions[(0, 1)])

    """
    array([ 3.9888620e-03,  3.6303997e-03,  6.0257912e-03, ...,
            -4.2915344e-06,  6.1130524e-04, -3.4993887e-01], dtype=float32)
    """
    # Group the effects by their indices along both axes.
    index_groupby = pd.DataFrame({"index_0": indices_list[0],
                                  "index_1": indices_list[1],
                                  "effects": effects}).groupby(["index_0", "index_1"])

    # Compute mean effects.
    mean_effects = index_groupby.mean()
    """
                      effects
    index_0 index_1          
    0       18       0.056193
            19       0.043221
    1       17       0.007076
            18       0.027295
    2       16       0.003114
            17       0.008690
            18       0.013942
    3       15       0.003245
            16       0.006434
    4       14       0.000753
            15       0.002147
    5       13       0.000112
            14       0.000088
        """

    # Get the indices of the mean values.
    group_indices = mean_effects.index
    """
    MultiIndex([( 0, 18),
            ( 0, 19),
            ( 1, 17),
            ( 1, 18),
            ( 2, 16),
            ( 2, 17),
            ( 2, 18),
            ( 3, 15),
            ( 3, 16),
            ( 4, 14),
            ( 4, 15),
           names=['index_0', 'index_1'])
    """

    valid_grid_indices = tuple(zip(*group_indices))
    """
    ((0, 0, 1, 1, 2, 2, 2, 3, ...), (18, 19, 17, 18, 16, 17, 18, 15, ...))
    """

    # Extract only the data.
    mean_effects = mean_effects.to_numpy().flatten()
    """
    array([ 5.61932288e-02,  4.32208516e-02,  7.07637845e-03,  2.72951536e-02, 3.11421417e-03,  8.68970156e-03, ...])
    """
    # Get the number of samples in each bin.
    n_samples = index_groupby.size().to_numpy()
    """
    array([502, 849, 818, 379, 476, 446, 214, 374,...])
    """

    # Create a 2D array of the number of samples in each bin.
    samples_grid = np.zeros(bins_list)  # samples_grid.shape == (20, 20)
    samples_grid[valid_grid_indices] = n_samples
    """
    array([[  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.,   0., 502., 849.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0., 818., 379.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0., 476., 446., 214.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0., 374., 433.,   0.,   0.,   0.],...])
    """
    ale = np.ma.MaskedArray(
        np.zeros((len(quantiles_list[0]), len(quantiles_list[1]))),
        mask=np.ones((len(quantiles_list[0]), len(quantiles_list[1]))),
    )  # ale.shape == (21, 21)

    # Mark the first row/column as valid, since these are meant to contain 0s.
    ale.mask[0, :] = False
    ale.mask[:, 0] = False

    # Place the mean effects into the final array.
    # Since `ale` contains `len(quantiles)` rows/columns the first of which are
    # guaranteed to be valid (and filled with 0s), ignore the first row and column.
    ale[1:, 1:][valid_grid_indices] = mean_effects

    # Record where elements were missing.
    missing_bin_mask = ale.mask.copy()[1:, 1:]  ## missing_bin_mask.shape == (20, 20)

    if np.any(missing_bin_mask):

        # Replace missing entries with their nearest neighbours.

        # Calculate the dense location matrices (for both features) of all bin centres.
        centres_list = np.meshgrid(*(_get_centres(quantiles) for quantiles in quantiles_list), indexing="ij")  # centres_list.shape == (2, 20, 20)
        """
        _get_centres(quantiles_list[0]) = array([-1.83455617e-03, -1.06035487e-03, -7.93087907e-04, -5.24551317e-04,
       -3.39353544e-04, -2.70835502e-04, -2.03874006e-04, -1.09371842e-04,
       -6.78721954e-05, -3.10543546e-05,  5.44617724e-05,  1.20644632e-04,
        1.87936545e-04,  2.56697276e-04,  2.92760364e-04,  3.58900827e-04,
        9.72093243e-04,  1.83610694e-03,  4.02675290e-03,  7.88283302e-03])
        """

        # Select only those bin centres which are valid (had observation).
        valid_indices_list = np.where(~missing_bin_mask)
        """
        (array([ 0,  0,  1,  1,  2,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  6,  7,
         7,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 12, 12, 12, 13, 13, 14,
        14, 15, 15, 16, 16, 17, 17, 18, 18, 19]),
         array([18, 19, 17, 18, 16, 17, 18, 15, 16, 14, 15, 13, 14,  9, 12, 13, 12,
                13, 10, 11,  9, 10, 11,  7,  8,  9,  7,  8,  4,  7,  9,  5,  6,  4,
                 6,  3,  5,  2,  3,  1,  2,  0,  1,  0]))
        """
        tree = cKDTree(
            np.hstack(
                tuple(
                    centres[valid_indices_list][:, np.newaxis]
                    for centres in centres_list
                )
            )
        )

        row_indices = np.hstack(
            [inds.reshape(-1, 1) for inds in np.where(missing_bin_mask)]
        )
        """
        row_indices = array([[ 0,  0],
           [ 0,  1],
           [ 0,  2],
           [ 0,  3],
           [ 0,  4],
           [ 0,  5],
           [ 0,  6],
           [ 0,  7],
           [ 0,  8],
           [ 0,  9],
           [ 0, 10],
           [ 0, 11],
           [ 0, 12],
           [ 0, 13],
           [ 0, 14],
           [ 0, 15],
           [ 0, 16],
           [ 0, 17],
           [ 1,  0],
           [ 1,  1],
           [ 1,  2],
           [ 1,  3],
           [ 1,  4],
           [ 1,  5],
           [ 1,  6],
           ...
        """
        # Select both columns for each of the rows above.
        column_indices = np.hstack(
            (
                np.zeros((row_indices.shape[0], 1), dtype=np.int8),
                np.ones((row_indices.shape[0], 1), dtype=np.int8),
            )
        )

        # Determine the indices of the points which are nearest to the empty bins.
        nearest_points = tree.query(tree.data[row_indices, column_indices])[1]  # nearest_points.shape == 356
        """
        array([ 0,  1,  2,  0,  4,  2,  0,  7,  4,  9,  7, 11,  9, 13, 14, 11, 14, 11,  0,  1, ...])
        """
        nearest_indices = tuple(
            valid_indices[nearest_points] for valid_indices in valid_indices_list
        )  # np.shape(nearest_indices) == (2, 356)

        # Replace the invalid bin values with the nearest valid ones.
        ale[1:, 1:][missing_bin_mask] = ale[1:, 1:][nearest_indices]  # ale.shape == (21, 21)

    # Compute the cumulative sums.
    ale = np.cumsum(np.cumsum(ale, axis=0), axis=1)

    # Subtract first order effects along both axes separately.
    for i in range(2):

        # Depending on `i`, reverse the arguments to operate on the opposite axis.
        flip = slice(None, None, 1 - 2 * i)

        # Undo the cumulative sum along the axis.
        first_order = ale[(slice(1, None), ...)[flip]] - ale[(slice(-1), ...)[flip]]

        # Average the diffs across the other axis.
        first_order = (
            first_order[(..., slice(1, None))[flip]]
            + first_order[(..., slice(-1))[flip]]
        ) / 2

        # Weight by the number of samples in each bin.
        first_order *= samples_grid

        # Take the sum along the axis.
        first_order = np.sum(first_order, axis=1 - i)

        # Normalise by the number of samples in the bins along the axis.
        first_order /= np.sum(samples_grid, axis=1 - i)

        # The final result is the cumulative sum (with an additional 0).
        first_order = np.array([0, *np.cumsum(first_order)]).reshape((-1, 1)[flip])

        # Subtract the first order effect.
        ale -= first_order

    # Compute the ALE at the bin centres.
    ale = (
        reduce(
            add,
            (
                ale[i : ale.shape[0] - 1 + i, j : ale.shape[1] - 1 + j]
                for i, j in list(product(*(range(2),) * 2))
            ),
        )
        / 4
    )

    # Centre the ALE by subtracting its expectation value.
    ale -= np.sum(samples_grid * ale) / train_set.shape[0]

    # Mark the originally missing points as such to enable later interpretation.
    ale.mask = missing_bin_mask

    return ale, quantiles_list


def ale_plot(
    model,
    train_set,
    features,
    bins=10,
    monte_carlo=False,
    predictor=None,
    features_classes=None,
    monte_carlo_rep=50,
    monte_carlo_ratio=0.1,
    rugplot_lim=1000,
    data_loader=None,
    color="black'",
    marker='x',
    markersize=1,
    exp=None,
    use_std=True,
    ale_std=None,
    ale_std_lower=None,
    ale_std_upper=None,
    folder_name="",
    type_of_output="speed",
    only_local_effects=False,
    linewidth=1
):
    """Plots ALE function of specified features based on training set.
    Parameters
    ----------
    model : object
        An object that implements a 'predict' method. If None, a `predictor` function
        must be supplied which will be used instead of `model.predict`.
    train_set : pandas.core.frame.DataFrame
        Training set on which model was trained.
    features : [2-iterable of] column label
        One or two features for which to plot the ALE plot.
    bins : [2-iterable of] int, optional
        Number of bins used to split feature's space. 2 integers can only be given
        when 2 features are supplied in order to compute a different number of
        quantiles for each feature.
    monte_carlo : boolean, optional
        Compute and plot Monte-Carlo samples.
    predictor : callable
        Custom prediction function. See `model`.
    features_classes : iterable of str, optional
        If features is first-order and a categorical variable, plot ALE according to
        discrete aspect of data.
    monte_carlo_rep : int
        Number of Monte-Carlo replicas.
    monte_carlo_ratio : float
        Proportion of randomly selected samples from dataset for each Monte-Carlo
        replica.
    rugplot_lim : int, optional
        If `train_set` has more rows than `rugplot_lim`, no rug plot will be plotted.
        Set to None to always plot rug plots. Set to 0 to always plot rug plots.
    data_loader : any, optional
        custom dataloader
    Raises
    ------
    ValueError
        If both `model` and `predictor` are None.
    ValueError
        If `len(features)` not in {1, 2}.
    ValueError
        If multiple bins were given for 1 feature.
    NotImplementedError
        If `features_classes` is not None.
    """

    if model is None and predictor is None:
        raise ValueError("If 'model' is None, 'predictor' must be supplied.")

    if features_classes is not None:
        raise NotImplementedError("'features_classes' is not implemented yet.")

    fig, ax = plt.subplots()

    features = _parse_features(features)

    if len(features) == 1:
        if not isinstance(bins, (int, np.integer)):
            raise ValueError("1 feature was given, but 'bins' was not an integer.")

        if features_classes is None:
            # Continuous data.

            if monte_carlo:
                """
                mc_replicates = np.asarray(
                    [
                        [
                            np.random.choice(range(train_set.shape[0]))
                            for _ in range(int(monte_carlo_ratio * train_set.shape[0]))
                        ]
                        for _ in range(monte_carlo_rep)
                    ]
                )
                for k, rep in enumerate(mc_replicates):
                    train_set_rep = train_set.iloc[rep, :]
                    # Make this recursive?
                    if features_classes is None:
                        # The same quantiles cannot be reused here as this could cause
                        # some bins to be empty or contain disproportionate numbers of
                        # samples.
                        mc_ale, mc_quantiles = _first_order_ale_quant(
                            model.predict if predictor is None else predictor,
                            train_set_rep,
                            features[0],
                            bins,
                            type_of_output=type_of_output,
                            only_local_effects=only_local_effects
                        )
                        _first_order_quant_plot(
                            ax, mc_quantiles, mc_ale, color="#1f77b4", alpha=0.06
                        )
                """
                raise NotImplementedError

            if use_std and not only_local_effects:
                ale, quantiles, ale_std_lower, ale_std_upper = _first_order_ale_quant(
                    model.predict if predictor is None else predictor,
                    train_set,
                    features[0],
                    bins,
                    use_std=True,
                    data_loader=data_loader,
                    type_of_output=type_of_output,
                    only_local_effects=only_local_effects)
            else:
                ale, quantiles = _first_order_ale_quant(
                    model.predict if predictor is None else predictor,
                    train_set,
                    features[0],
                    bins,
                    use_std=False,
                    data_loader=data_loader,
                    type_of_output=type_of_output,
                    only_local_effects=only_local_effects
                )

            _ax_labels(ax, "Feature '{}'".format(features[0]), "")
            _ax_title(ax,
                      "First-order ALE of feature '{0}'".format(features[0]),
                      "Bins : {0} - Monte-Carlo : {1}".format(len(quantiles) - 1, "False",),)
            ax.grid(True, linestyle="-", alpha=0.4)

            if rugplot_lim is None or train_set.shape[0] <= rugplot_lim:
                sns.rugplot(train_set[features[0]], ax=ax, alpha=0.2)

            _first_order_quant_plot(ax, quantiles, ale,
                                    exp=exp, feature=features[0], folder_name=folder_name,
                                    ale_std_lower=ale_std_lower, ale_std_upper=ale_std_upper,
                                    color=color, marker=marker, markersize=markersize, linewidth=linewidth)
            _ax_quantiles(ax, quantiles)

            if exp and features[0] and folder_name:
                uuid_str = str(uuid.uuid4())[:4]
                create_folder_if_doesnt_exist(exp.path_to_figures+folder_name, _raise=False, verbose=True)
                np.save(exp.path_to_figures + folder_name + f'/{features[0]}_centres_{uuid_str}.npy',
                        _get_centres(quantiles))
                np.save(exp.path_to_figures + folder_name + f'/{features[0]}_quantiles_{uuid_str}.npy', quantiles)
                np.save(exp.path_to_figures + folder_name + f'/{features[0]}_ale_{uuid_str}.npy', ale)
                if use_std:
                    np.save(exp.path_to_figures + folder_name + f'/{features[0]}_ale_std_lower_{uuid_str}.npy',
                            ale_std_lower)
                    np.save(exp.path_to_figures + folder_name + f'/{features[0]}_ale_std_upper_{uuid_str}.npy',
                            ale_std_upper)

    elif len(features) == 2:
        if features_classes is None:
            # Continuous data.
            ale, quantiles_list = _second_order_ale_quant(
                model.predict if predictor is None else predictor,
                train_set,
                features,
                bins,
                data_loader=data_loader,
                type_of_output=type_of_output
            )
            _second_order_quant_plot(fig, ax, quantiles_list, ale)
            _ax_labels(
                ax,
                "Feature '{}'".format(features[0]),
                "Feature '{}'".format(features[1]),
            )
            for twin, quantiles in zip(("x", "y"), quantiles_list):
                _ax_quantiles(ax, quantiles, twin=twin)
            _ax_title(
                ax,
                "Second-order ALE of features '{0}' and '{1}'".format(
                    features[0], features[1]
                ),
                "Bins : {0}x{1}".format(*[len(quant) - 1 for quant in quantiles_list]),
            )
    else:
        raise ValueError(
            "'{n_feat}' 'features' were given, but only up to 2 are supported.".format(
                n_feat=len(features)
            )
        )
    plt.show()
    return ax
