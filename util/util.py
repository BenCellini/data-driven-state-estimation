import numpy as np
import pandas as pd
import scipy
import copy
from scipy import interpolate
from collections.abc import Iterable


def wrapToPi(rad):
    rad_wrap = copy.copy(rad)
    q = (rad_wrap < -np.pi) | (np.pi < rad_wrap)
    rad_wrap[q] = ((rad_wrap[q] + np.pi) % (2 * np.pi)) - np.pi
    return rad_wrap


def wrapTo2Pi(rad):
    rad = copy.copy(rad)
    rad = rad % (2 * np.pi)
    return rad


def hampel_filter(data, window_size, n_sigmas=3):
    """
    Applies the Hampel filter to a pandas Series.

    Args:
        data (pd.Series): The input time series data.
        window_size (int): The size of the rolling window.
        n_sigmas (float): The number of MADs to use as the outlier threshold.

    Returns:
        pd.Series: The filtered time series data with outliers replaced by the median.
    """

    if not isinstance(data, pd.Series):
        data = pd.Series(data)

    k = 1.4826  # Adjustment factor for converting MAD to standard deviation

    rolling_median = data.rolling(window=window_size, center=True).median()

    rolling_mad = k * abs(data - rolling_median).rolling(window=window_size, center=True).median()

    outlier_indices = abs(data - rolling_median) > (n_sigmas * rolling_mad)

    data[outlier_indices] = rolling_median[outlier_indices]

    return data


# def log_interp1d(xx, yy, kind='linear'):
#     logx = np.log10(xx)
#     logy = np.log10(yy)
#     lin_interp = interpolate.interp1d(logx, logy, kind=kind)
#     log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
#     return log_interp

def log_interpolator(time, values, kind='linear'):
    """
    Creates an interpolation function for 1D time-series data with logarithmically scaled values.

    Parameters:
    - time (array-like): Evenly spaced time points.
    - values (array-like): Logarithmically scaled values corresponding to time points.
    - kind (str): Interpolation type ('linear', 'quadratic', 'cubic', etc.).

    Returns:
    - function: Interpolation function that takes time as input and returns interpolated values.
    """
    log_values = np.log(values)  # Convert to log-space
    interp_func = scipy.interpolate.interp1d(time, log_values, kind=kind, fill_value='extrapolate')  # Interpolate in log-space

    return lambda t_new: np.exp(interp_func(t_new))  # Return function that converts back to original scale


def segment_from_df(df, column='time', val=0.0, label_index=False):
    """ Pulls out segments from data frame based on where 'val' shows up in 'column'.
    """

    # Fins start of segments
    segment_start = np.where(df[column].values == val)[0].squeeze()  # where segments start
    n_segment = segment_start.shape[0]  # number of segments

    segment_list = []  # list of individual segments
    for n in range(n_segment):
        if n == (n_segment - 1):
            segment_end = df.shape[0]
        else:
            segment_end = segment_start[n + 1]

        segment = df.iloc[segment_start[n]:segment_end, :]

        if label_index:
            segment.insert(0, 'traj_index', (n * np.ones(segment.shape[0])).astype(int))

        segment_list.append(segment)

    return segment_list, n_segment


def augment_dataframe(df, aug_column_names=None, keep_column_names=None, w=1, direction='backward', remove_nans=False):
    """
    Augments data by collecting prior or future rows into new columns for each specified column.

    :param df: pandas.DataFrame
        The input data frame containing the columns to be augmented.

    :param aug_column_names: list, optional
        List of column names to augment. If None, all columns are augmented.

    :param keep_column_names: list, optional
        List of column names to keep without augmentation. If None, no columns are kept.

    :param w: int, optional, default=1
        The window size for collecting prior or future rows. Determines how many rows to look back or forward.

    :param direction: {'backward', 'forward'}, optional, default='backward'
        Specifies the direction for collecting the rows:
        - 'backward' to collect previous rows.
        - 'forward' to collect future rows.

    :return: pandas.DataFrame
        The augmented data frame with new columns named as `<original_column_name>_<index>` for each augmented column.
        Non-augmented columns are added if specified.

    :raises Exception: If `direction` is not one of 'forward' or 'backward'.
    """

    df = df.reset_index(drop=True)

    # Default for testing
    if df is None:
        df = np.atleast_2d(np.arange(0, 11, 1, dtype=np.double)).T
        df = np.matlib.repmat(df, 1, 4)
        df = pd.DataFrame(df, columns=['a', 'b', 'c', 'd'])
        aug_column_names = ['a', 'b']
    else:
        if aug_column_names is None:
            aug_column_names = df.columns

    n_row = df.shape[0]
    n_row_train = n_row - w + 1

    # Initialize the dictionary for augmented data
    df_aug_dict = {}

    # Create new column names and prepare data matrices in a vectorized way
    for a in aug_column_names:
        # Preallocate the augmented data matrix
        df_aug_dict[a] = np.full((n_row_train, w), np.nan)

        # Vectorize the column augmentation
        for i in range(w):
            if direction == 'backward':
                df_aug_dict[a][:, i] = df[a].iloc[w - 1 - i:n_row - i].values
            elif direction == 'forward':
                df_aug_dict[a][:, i] = df[a].iloc[i:n_row - w + 1 + i].values
            else:
                raise Exception("direction must be 'forward' or 'backward'")

    # Convert the augmented data matrices to DataFrames and assign proper column names
    df_aug = pd.concat([pd.DataFrame(df_aug_dict[a], columns=[f"{a}_{i}" for i in range(w)]) for a in aug_column_names],
                       axis=1)

    # Add non-augmented data columns if specified
    if keep_column_names is not None:
        for c in keep_column_names:
            if direction == 'backward':
                df_aug[c] = df[c].iloc[w - 1:n_row].reset_index(drop=True)
            elif direction == 'forward':
                df_aug[c] = df[c].iloc[0:n_row - w].reset_index(drop=True)
            else:
                raise Exception("direction must be 'forward' or 'backward'")

    # Optionally remove rows with NaN values
    if remove_nans:
        df_aug = df_aug.dropna()

    return df_aug


def augment_columns(df, augment_columns_names=None, n=4, direction='forward', remove_nans=False):
    """
    Augments specified columns by shifting values either forward, backward, or centered.

    :param df: pandas DataFrame
        The input DataFrame containing the data.

    :param augment_columns_names: list of str
        List of column names to augment.

    :param n: int
        The number of steps (how many shifts to create).

    :param direction: str, optional, default 'forward'
        The direction to shift the values. It can be:
        - 'forward': shifts the values down (past values come in front).
        - 'backward': shifts the values up (future values come in front).
        - 'centered': shifts the values symmetrically around the original value.

    :param remove_nans: bool, optional, default False
        Whether to remove rows containing NaN values after augmentation.

    :returns: pandas DataFrame
        The DataFrame with the augmented columns.

    :raises ValueError: If the direction parameter is not one of 'forward', 'backward', or 'centered'.
    """

    augmented_df = df.copy()

    for col in augment_columns_names:
        for i in range(n):
            if direction == 'forward':
                augmented_df[f'{col}_{i}'] = augmented_df[col].shift(-i)
            elif direction == 'backward':
                augmented_df[f'{col}_{i}'] = augmented_df[col].shift(i)
            elif direction == 'centered':
                augmented_df[f'{col}_{i}'] = augmented_df[col].shift(i - (n+1) // 2)
            else:
                raise ValueError("Direction must be 'forward', 'backward', or 'centered'")

    # Optionally remove rows with NaN values
    if remove_nans:
        augmented_df = augmented_df.dropna()

    return augmented_df


def get_sequential_inputs(df=None, names=None, delimiter='_', time_steps=None):
    """ Collect the sequentially named columns from data-frame.
        ex: names = ('a', 'b') ; delimiter = '_'; time_steps = n
        df.columns = a_0, a_1, ..., a_n, b_0, b_1, ..., b_n

    :param pandas.DataFrame df: pandas data-frame
    :param list | tuple names: string that column names start with
    :param str delimiter: string the separates name from numbered index
    :param int | iterable time_steps: how many (int) or what (iterable) time-steps to use.
    If None, use all time-steps available

    :return pandas.DataFrame X: curated data-frame
    """
    X = []
    for i in names:  # get columns for each name
        if time_steps is not None:
            if isinstance(time_steps, Iterable):  # use the time-steps in iterable
                selected_columns = []
                for t in time_steps:  # each time-step
                    selected_columns.append(np.where(df.columns.str.startswith(i + delimiter + str(t)))[0])

                selected_columns = np.hstack(selected_columns)

            elif isinstance(time_steps, int):  # use the first n time-steps
                selected_columns = np.where(df.columns.str.startswith(i + delimiter))[0]
                if time_steps > selected_columns.shape[0]:
                    raise Exception('time_steps do not exist in input data-frame')
                else:
                    selected_columns = selected_columns[0:time_steps]
            else:
                raise Exception('time_steps must be an int or iterable')
        else:  # use all time-steps
            selected_columns = np.where(pd.Series(df.columns).str.startswith(i + delimiter))[0]

        # Get columns
        x = df.iloc[:, selected_columns]
        X.append(x)

    # Concatenate column for each name
    X = pd.concat(X, axis=1)

    return X


def get_indices(fisher_data_structure, states_list=None, sensors_list=None, time_steps_list=None):
    """Get indices in data structure corresponding to states, sensors, & time-steps.
    """

    data = fisher_data_structure
    index_map = np.NaN * np.zeros((len(sensors_list), len(states_list), len(time_steps_list)))
    n_cond = len(data['states'])
    for j in range(n_cond):
        for n, states in enumerate(states_list):
            if states == data['states'][j]:
                for p, sensors in enumerate(sensors_list):
                    if sensors == data['sensors'][j]:
                        for k, time_steps in enumerate(time_steps_list):
                            if time_steps == data['time_steps'][j]:
                                index_map[p, n, k] = j

    # Check
    if np.sum(np.isnan(index_map)) > 0:
        # print('Input states, sensors, or time-steps that do not exist')
        raise Exception('Input states, sensors, or time-steps that do not exist')
    else:
        index_map = index_map.astype(int)

    n_sensors, n_states, n_time_steps = index_map.shape

    return index_map, n_sensors, n_states, n_time_steps
