import pandas as pd
import numpy as np
import h5py


def load_csv(path):
    """
    Load EEG data from a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded data.
    """
    return pd.read_csv(path)



# extract eeg with number of channel
def extract_eeg(
  df,
  n_channels=16,
  channels=None
) -> np.ndarray:
    """
    Extract EEG channels ch1 ... chN from a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        EEG recording data.
    n_channels : int
        Number of EEG channels.
    channels : list[str] or None
        Specific channel names to extract.

    Returns
    -------
    np.ndarray
        Time array with shape (n_sample, )
        EEG array with shape (n_channels, n_samples).
    """
    if channels is None:
        channel_names = [
            f"ch{ii}" for ii in range(1, n_channels + 1)
        ]
    else:
        channel_names = channels

    t = df["t"].to_numpy()
    eeg = df[channel_names].to_numpy().T

    return t, eeg

# Load data from matlab. After v7.3 .mat using h5py for loading data
def extract_eeg(data: np.ndarray, n_channels: int = 16):
    """
    Extract time, EEG channels, and trigger from loaded data.

    Parameters
    ----------
    data : np.ndarray
        Loaded recording data.
    n_channels : int
        Number of EEG channels.

    Returns
    -------
    t : np.ndarray
        Time vector.

    eeg : np.ndarray
        EEG data with shape (n_channels, n_samples).

    trig : np.ndarray
        Trigger signal.
    """

    t = data[:, 0]

    eeg = data[:, 1:n_channels + 1].T

    trig = data[:, -1]

    return t, eeg, trig





