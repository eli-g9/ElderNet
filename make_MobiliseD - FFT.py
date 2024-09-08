import os
import pickle
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import mode
from tqdm import tqdm


### CONSTANTS ###
RESAMPLED_HZ = 30
WINDOW_SEC = 10  # seconds
WINDOW_OVERLAP_SEC = 0  # seconds
WINDOW_LEN = int(RESAMPLED_HZ * WINDOW_SEC)
WINDOW_OVERLAP_LEN = int(RESAMPLED_HZ * WINDOW_OVERLAP_SEC)
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN
DAY_MAPPING = {1:1, 2:4}
REV_DAY_MAPPING = {1:1, 4:2}
LABELED_DAYS = [1, 4]
ORIGINAL_FS = 50
NUM_BINS_FFT = 10


def average_fft_in_bins(fft_output, freqs):
    """
    Compute the average and median FFT values in bins for the input fft_output and freqs.

    Parameters
    ----------
    fft_output : numpy.ndarray
        A 3D numpy array containing acceleration data for three axes (X, Y, Z) and time.
        The shape of the array should be (number_of_samples, 3).

    freqs : numpy.ndarray
        An array containing the corresponding frequencies for the input fft_output.

    NUM_BINS_FFT : int, optional
        The number of bins to use for computing the average and median FFT values. The default is 10.

    Returns
    -------
    avg_fft_values, median_fft_values : numpy.ndarray
        Two numpy arrays containing the average and median FFT values in bins, respectively.
        The shape of the arrays should be (number_of_bins,).
    """
    # Determine the frequency range
    min_freq = np.min(freqs)
    max_freq = np.max(freqs)

    # Create bins
    bins = np.linspace(min_freq, max_freq, NUM_BINS_FFT + 1)

    # Compute average FFT values in each bin
    avg_fft_values = []
    median_fft_values = []
    for i in range(len(fft_output)):
        avg_values, median_values = [], []
        for j in range(NUM_BINS_FFT):
            bin_mask = (freqs >= bins[j]) & (freqs < bins[j + 1])
            avg_values.append(np.mean(np.abs(fft_output[i][bin_mask])))
            median_values.append(np.median(np.abs(fft_output[i][bin_mask])))
        avg_fft_values.append(avg_values)
        median_fft_values.append(median_values)

    return avg_fft_values, median_fft_values


def fft_bins(data, data_freq):
    """
    Compute the Fast Fourier Transform (FFT) for each acceleration axis of the input data,
    and then compute the average and median FFT values in bins.

    Parameters
    ----------
    data : numpy.ndarray
        A numpy array containing acceleration data for three axes (X, Y, Z) and time.
        The shape of the array should be (number_of_samples, 3).

    data_freq : float
        The sampling frequency of the input data.

    Returns
    -------
    avg_fft_values, median_fft_values : numpy.ndarray
        Two numpy arrays containing the average and median FFT values in bins, respectively.
        The shape of the arrays should be (number_of_bins,).
    """
    # Compute FFT for each acceleration axis
    fft_output = [np.fft.fft(data[:, i]) for i in range(3)]

    # Compute the corresponding frequencies
    n = data.shape[0]
    freqs = np.fft.fftfreq(n, d=1/data_freq)

    # Compute the average and median FFT values in bins
    avg_fft_values, median_fft_values = average_fft_in_bins(fft_output, freqs)

    return np.array([avg_fft_values, median_fft_values])


def resample_data(data, labels, original_fs, target_fs):
    '''
    :param data: Numpy array. Data to resample.
    :param original_fs: Float, the raw data sampling rate
    :param target_fs: Float, the sampling rate of the resampled signal
    :return: resampled data
    '''
    # calculate resampling factor
    resampling_factor = original_fs / target_fs
    # calculate number of samples in the resampled data and labels
    num_samples = int(len(data) / resampling_factor)
    # use scipy.signal.resample function to resample data, labels, and subjects
    resampled_data = signal.resample(data, num_samples)

    label_indices = np.linspace(0, len(labels) - 1, num_samples).astype(int)
    aligned_labels = labels[label_indices]

    return resampled_data, aligned_labels


def convert_data(task_data):
    # GENEActiv_X GENEActiv_Y GENEActiv_Z GENEActiv_Magnitude
    return task_data[['GENEActiv_X', 'GENEActiv_Y', 'GENEActiv_Z', 'label']].values


def convert_labels(labels):
    y_one_hot = np.eye(5)[labels.astype(int)]
    return y_one_hot


def data_to_windows(data, labels):
    num_windows = len(labels) // WINDOW_STEP_LEN
    data_win = np.empty((num_windows, WINDOW_LEN, 3))
    labels_win = np.empty((num_windows,))
    fft_features = np.empty((num_windows, 2 * 3 * NUM_BINS_FFT))  # 3 axes * NUM_BINS_FFT bins

    for i, start_idx in enumerate(range(0, (len(labels) - WINDOW_STEP_LEN), WINDOW_STEP_LEN)):
        end_idx = start_idx + WINDOW_LEN
        window_data = data[start_idx:end_idx]
        data_win[i, :, :] = window_data
        labels_win[i] = mode(labels[start_idx:end_idx])[0]

        # Compute FFT for the current window
        fft_out = fft_bins(window_data, RESAMPLED_HZ)
        fft_out = fft_out.reshape(2 * 3 * NUM_BINS_FFT)
        fft_features[i, :] = fft_out

    return data_win, labels_win, fft_features


def times_to_overlaps(tasks_times, window_len, threshold=0):
    """
    window_len - length of window in seconds
    threshold - float in range [0, 1] - decide if we use the partial window.
                0 is all partial windows, 1 is none.
                0.5 for window_len=10 will take 5s and longer.
    """
    new_times = []
    for start, end in tasks_times:
        task_len = end - start
        # Few tasks are less than window length, we will ignore them
        if task_len > window_len:
            remainder = task_len % window_len
            regular_task = (start, end - remainder)
            if remainder != 0 and ((remainder / window_len) >= threshold):
                last_window = (end - window_len, end)
                new_times.append(regular_task)
                new_times.append(last_window)
            else:
                new_times.append(regular_task)
    return new_times


def get_overlap_time(task_data, tasks_times):
    new_times = np.array([])
    init = False
    for start, end in tasks_times:
        data = convert_data(
            task_data[
                (task_data['timestamp'] >= start) & 
                (task_data['timestamp'] <= end)])
        if not init:
            new_times = data
            init = True
        else:
            new_times = np.vstack((new_times, data))
    return new_times


def binary_one_hot(labels):
    # Converts [1, 0, 0, 0, 0] -> [1, 0], [0, 1, 0, 0, 0] -> [0, 1]
    first = labels[:, 0]
    output_array = np.zeros((labels.shape[0], 2))
    output_array[first == 1, 1] = 1
    output_array[first != 1, 0] = 1
    return output_array


def get_data_directory():
    try:
        work_directory = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        work_directory = os.getcwd()
    download_path = os.path.join(work_directory, 'Data')
    return work_directory, download_path


def get_all_subjects(key_data):
    return sorted(list(set(key_data["subject_id"])))


def get_tasks(key_data, subject_id, visit, task_filter=None):
    """
    Returns (visits, tasks_times, labels)

    """
    cols = ["visit", "timestamp_start", "timestamp_end", "score"]

    subject_data = key_data[
        (key_data["subject_id"] == subject_id) &
        (key_data["body_segment"] == "RightUpperLimb") &
        (~key_data["task_code"].isin(['drawg', 'ftnl', 'raml'])) &
        (key_data["phenotype"] == "tremor") &
        (key_data["visit"] == visit)
    ][cols]

    if task_filter:
        subject_data = subject_data[key_data["task_id"] == task_filter]

    return (zip(subject_data["timestamp_start"], 
                subject_data["timestamp_end"]),  
                subject_data["score"])


def convert_id_to_foldername(subject_id):
    naming_dict = {"NYC": "_NY",
                   "BOS": ""}
    split_id = subject_id.split("_")
    number = split_id[0]
    city = split_id[1]
    return f"patient{number}{naming_dict[city]}"
    

def get_subject_data(subject_id, day, data_dir):
    """
    Returns a pandas dataframe with subject data from day
    """
    folder = convert_id_to_foldername(subject_id)
    patient_data_dir = os.path.join(data_dir, folder)
    filepath = os.path.join(patient_data_dir, f"rawdata_day{day}.txt")
    return pd.read_csv(filepath, delimiter="\t", low_memory=False)
    

# Optionally make list_of_times only for the correct day
def get_data_from_day(subject_id, day, data_dir, list_of_times=None, labels=None):
    day_data = get_subject_data(subject_id, day, data_dir)
    if list_of_times == None:
        return day_data # timestamp GENEActiv_X GENEActiv_Y GENEActiv_Z GENEActiv_Magnitude
    day_data['during_task'] = False
    day_data['label'] = 0  # Add a new column for labels
    for (task_start, task_end), label in zip(list_of_times, labels):
        # if day_start <= task_start <= day_end:
        start_index = np.searchsorted(day_data['timestamp'], task_start)
        end_index = np.searchsorted(day_data['timestamp'], task_end)
        day_data.loc[start_index:end_index, 'during_task'] = True
        day_data.loc[start_index:end_index, 'label'] = int(label)

    return day_data[day_data['during_task'] == True]


def data_to_pickle(key_data, days=[1, 4], threshold=0, binary_labels=True, task_filter=None):
    _, data_dir = get_data_directory()
    data_total = np.array([])
    fft_total = np.array([])
    labels_total = np.array([])
    subjects_total = np.array([])
    data_file = os.path.join(data_dir, "WindowsData.p")
    fft_file = os.path.join(data_dir, "WindowsFFT.p")
    labels_file = os.path.join(data_dir, "WindowsLabels.p")
    groups_file = os.path.join(data_dir, "WindowsSubjects.p")
    subjects = get_all_subjects(key_data)
    totals_init = False
    for subject_id in tqdm(subjects):
        for day in days:
            if day in LABELED_DAYS:
                tasks_times, labels = get_tasks(key_data, subject_id, REV_DAY_MAPPING[day], task_filter)
                tasks_times = list(tasks_times)
                overlap_tasks_times = times_to_overlaps(tasks_times, WINDOW_SEC, threshold=threshold)
                task_data = get_data_from_day(subject_id, day, data_dir, tasks_times, labels)
                overlap_task_data = get_overlap_time(task_data, overlap_tasks_times)
                overlap_data = overlap_task_data[:, :-1]
                overlap_labels = overlap_task_data[:, -1]
                resampled_data, labels = resample_data(overlap_data, overlap_labels, ORIGINAL_FS, RESAMPLED_HZ)
                X = resampled_data
                Y = labels.astype('float').round().ravel()
                data_win, labels_win, fft_win = data_to_windows(X, Y)
                # I don't know why but this is necessary for the shape to be the same.
                data_win = data_win.reshape(data_win.shape[0], data_win.shape[-1], data_win.shape[1])
                one_hot_labels = convert_labels(abs(labels_win).round())
                subjects_win = [subject_id] * len(labels_win)
                if totals_init:
                    data_total = np.vstack((data_total, data_win))
                    fft_total = np.vstack((fft_total, fft_win))
                    labels_total = np.vstack((labels_total, one_hot_labels))
                    subjects_total = np.hstack((subjects_total, subjects_win))
                else:
                    data_total = data_win
                    fft_total = fft_win
                    labels_total = one_hot_labels
                    subjects_total = subjects_win
                    totals_init = True

    if binary_labels:
        labels_total = binary_one_hot(labels_total)

    with open(data_file, 'wb') as f:
        pickle.dump(data_total, f)
    with open(fft_file, 'wb') as f:
        pickle.dump(fft_total, f)
    with open(labels_file, 'wb') as f:
        pickle.dump(labels_total, f)
    with open(groups_file, 'wb') as f:
        # needs to look like [SUBJ1, SUBJ1, ...] as number of tasks
        pickle.dump(subjects_total, f)


def main():
    work_directory, download_path = get_data_directory()
    keydata_file = os.path.join(download_path, 'key_data.csv')
    key_data = pd.read_csv(keydata_file)
    data_to_pickle(key_data, days=[1, 4], threshold=0, binary_labels=True)


if __name__ == "__main__":
    main()
