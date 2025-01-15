import pathlib
import numpy as np
from typing import List, Optional

from utils import get_audio_files_from_dir, get_audio_file_data,\
    write_pickle, split_files, extract_features


def main(dataset_root_path: pathlib.Path, dataset_output_path: pathlib.Path,
         genres: List[str]):
    """Extracting 30sec samples of songs from all used genres from given path.
    The songs are then split into training, testing and validation sets and stored.

    :param dataset_root_path: rooth path of the dataset
    :type dataset_root_path: List[pathlib.Path]
    :param dataset_output_path: output for the dataset
    :type dataset_output_path: pathlib.Path
    :param genres: Genres the dataset is split into
    :type genres: list[str]
    """

    # Mapping from genres to integer labeling
    genre_to_int_label = {genre: i for i, genre in enumerate(genres)}

    # Go through each genre
    for genre in genres:
        genre_path = dataset_root_path / genre
        audio_files = get_audio_files_from_dir(genre_path)

        print(f"Processing {genre} songs.")

        # Split files into training, testing and validation files
        data_splits = split_files(audio_files, 0.5, 0.2, 0.3)

        # Extract features from every file
        for split, audio_files in data_splits.items():

            print(f" - Processing {split} files.")

            output_dir = pathlib.Path(dataset_output_path / split)
            output_dir.mkdir(exist_ok=True)

            for file in audio_files:
                
                # Get the data from 'audio_file'
                audio_data, sr = get_audio_file_data(file)

                # Extract spectrogram for the audio data
                spectrograms = extract_features(audio_data, sr)

                for i in range(len(spectrograms)):
                    # Save the spectrogram into a file
                    output_data = (genre_to_int_label[genre], spectrograms[i])
                    output_filename = output_dir / f"{file.name}_{i}.pickle"
                    write_pickle(output_filename, output_data)


if __name__ == '__main__':
    # Dataset paths
    data_set_root_path = pathlib.Path("Data/genres_original/")
    dataset_output_path = pathlib.Path("Data/")
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    main(data_set_root_path, dataset_output_path, genres)
