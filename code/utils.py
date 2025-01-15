from typing import List, Union, Any, Tuple, Dict, Optional
import pathlib
import pickle
import numpy as np
import librosa
import random


def split_files(audio_files: List[pathlib.Path], train: float,
                validation: float, testing: float) \
        -> Dict[str, List[pathlib.Path]]:
    """Splits given list of files into training, validation and testing files.
        The values are percentages.

    :param testing: split size for testing
    :param train:  split size for training
    :param validation: split size for validation
    :param audio_files: List of files to be split
    :return: returns a dictionary with splits as keys and files as values
    """
    if (train + validation + testing) == 0:
        raise ValueError("Split percentages must add up to 1.")

    random.shuffle(audio_files)

    tot_files = len(audio_files)
    train_end = int(train * tot_files)
    validation_end = train_end + int(validation * tot_files)

    train_files = audio_files[:train_end]
    validation_files = audio_files[train_end:validation_end]
    testing_files = audio_files[validation_end:]

    data_splits = {'training': train_files,
                   'validation': validation_files,
                   'testing': testing_files}
    return data_splits


def min_max_normalize(spectrogram: np.ndarray) -> np.ndarray:
    """Normalizes spectrogram with min-max normalization.

    :param spectrogram: Spectrogram to be normalized
    :return: Normalized spectrogram
    """

    min_val = np.min(spectrogram)
    max_val = np.min(spectrogram)

    if max_val - min_val > 0:
        return (spectrogram - min_val) / (max_val - min_val)
    return spectrogram


def extract_features(signal: np.ndarray,
                     sampling_rate: int,
                     window: Optional[str] = 'hamming',
                     n_mels : Optional[int] = 64) \
        -> np.ndarray:
    """Extracts a log-mel spectrogram of the given signal and cuts it down
    to 3 second long segments. The final segment is padded with zeros to reach needed length.

    :param signal: Audio signal
    :type signal: numpy.ndarray
    :param sampling_rate: Sampling rate of the signal
    :type sampling_rate: int
    :param window: Window type (default 'hamming')
    :type window: Optional[str]
    :return: List of numpy arrays, each being a 3 second length snippet of the
    log-mel spectrogram, with the final snippet zero-padded if shorter than 3 seconds.
    """
    # 23ms window
    n_fft = int(sampling_rate * 0.023)

    # 50% overlap
    hop_length = n_fft // 2

    S = librosa.feature.melspectrogram(y=signal,
                                       sr=sampling_rate,
                                       n_fft=n_fft,
                                       hop_length=hop_length,
                                       window=window,
                                       n_mels=n_mels)

    frame_duration = hop_length / sampling_rate
    frames_per_3_sec = int(3 / frame_duration)

    segments = []

    num_segments = S.shape[1] // frames_per_3_sec

    # Extracting each full 3 sec segment of the spectrogram
    for i in range(num_segments):
        start_frame = i *frames_per_3_sec
        end_frame = start_frame + frames_per_3_sec
        segment = S[:, start_frame:end_frame]
        segments.append(segment)

    # Check for remainder and pad remainder with zeros if necessary
    remainder = S.shape[1] % frames_per_3_sec
    if remainder > 0:
        last_segment = S[:, -remainder:]
        pad_length = frames_per_3_sec - remainder
        # Padding to match the 3 second length
        padded_segment = np.pad(last_segment,
                                ((0, 0),
                                 (0, pad_length)),
                                mode='constant',
                                constant_values=0)
        segments.append(padded_segment)

    return segments


def get_audio_files_from_dir(dir_name: Union[str, pathlib.Path]) \
        -> List[pathlib.Path]:
    """Returns the audio files in the given directory using pathlib.

    :param dir_name: Name of the directory.
    :type dir_name str
    :return: Filenames in the directory
    :rtype: list[pathlib.Path]
    """
    return list(pathlib.Path(dir_name).iterdir())


def get_audio_file_data(audio_file: Union[str, pathlib.Path]) \
        -> Tuple[np.ndarray, float]:
    """Returns the data from the 'audio_file'

    :param audio_file: Path of the audio file
    :type audio_file: str
    :return: Data of the 'audio_file'
    :rtype: Tuple[np.ndarray, float]
    """
    return librosa.core.load(path=audio_file, sr=None, mono=False)


def write_pickle(filename: Union[str, pathlib.Path], data: Any):
    """Saves the given file as a pickle file

    :param filename: filename for the data
    :param data: Data to be store
    """
    with pathlib.Path(filename).open('wb') as pkl_file:
        pickle.dump(data, pkl_file)
