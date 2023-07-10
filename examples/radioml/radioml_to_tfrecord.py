import tensorflow as tf
import tensorflow_datasets as tfds
from smart_tfrecord_writer import Writer
import h5py
import numpy as np


class RadioMLWriter(Writer):
    def extend_meta_data(self):
        """Optional method that allows additional information to be stored in meta data.

        Returns
        -------
        tuple
            Components for the description, homepage, supervised keys, and citation.
        """
        description = """
        A dataset which includes both synthetic simulated channel effects of 24 digital 
        and analog modulation types which has been validated.  This dataset was used in 
        the paper Over-the-air deep learning based radio signal classification which was
        published in 2017 in IEEE Journal of Selected Topics in Signal Processing, which 
        provides additional details and description of the dataset."""

        homepage = "https://www.deepsig.ai/datasets"
        supervised_keys = ("rf_signal", "label")
        citation = """
        @ARTICLE{8267032,
            author={O’Shea, Timothy James and Roy, Tamoghna and Clancy, T. Charles},
            journal={IEEE Journal of Selected Topics in Signal Processing}, 
            title={Over-the-Air Deep Learning Based Radio Signal Classification}, 
            year={2018},
            volume={12},
            number={1},
            pages={168-179},
            doi={10.1109/JSTSP.2018.2797022}}
        """

        return description, homepage, supervised_keys, citation

    def features(self):
        """Required function that defines the structure of a single example.py

        Returns
        -------
        FeaturesDict
            Description of the features in the dataset.
        """
        features = tfds.features.FeaturesDict(
            {
                "rf_signal": tfds.features.Tensor(
                    shape=(1024, 2),
                    dtype=np.float32,
                    doc="A radio signal with I and Q components.",
                ),
                "label": tfds.features.ClassLabel(
                    names=[
                        "OOK",
                        "4ASK",
                        "8ASK",
                        "BPSK",
                        "QPSK",
                        "8PSK",
                        "16PSK",
                        "32PSK",
                        "16APSK",
                        "32APSK",
                        "64APSK",
                        "128APSK",
                        "16QAM",
                        "32QAM",
                        "64QAM",
                        "128QAM",
                        "256QAM",
                        "AM-SSB-WC",
                        "AM-SSB-SC",
                        "AM-DSB-WC",
                        "AM-DSB-SC",
                        "FM",
                        "GMSK",
                        "OQPSK",
                    ],
                ),
                "snr": tfds.features.Scalar(
                    dtype=np.float32, doc="Average SNR of the signal."
                ),
            }
        )

        return features

    def process_data(self, index):
        parsed_instance = {}

        with h5py.File(self.source_directory, "r") as f:
            rf_signal = f["X"][index]
            label = f["Y"][index]
            snr = f["Z"][index]

            parsed_instance[
                "rf_signal"
            ] = rf_signal  # Generic Tensor feature expects shape of (1024, 2)
            parsed_instance["label"] = np.argmax(
                label
            )  # ClassLabel feature expects single integer value
            parsed_instance["snr"] = np.squeeze(
                snr.astype(np.float32)
            )  # Scalar feature expects a single value

        return parsed_instance


if __name__ == "__main__":
    # Source and destination files
    # TODO: Add paths to the source and destination files
    src_path = "/path/to/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5"
    dest_path = "/path/to/radio_ml_2018.01"
    writer = RadioMLWriter(
        source_directory=src_path,
        destination_directory=dest_path,
    )

    # Example indices
    train_indexes = list(range(3_000))
    test_indexes = list(range(1_000))

    # Structure splits information for the writer into a list of dictionaries
    splits_info = [
        {"name": "train", "info": train_indexes},
        {"name": "test", "info": test_indexes},
    ]

    # Write the records and have approximately 250MB per shard
    writer.write_records(
        splits_info=splits_info,
        mb_per_shard=250,
        n_estimates_mb_per_example=1,
        shuffle=True,
        verbose=3,
    )

    # Use TFDS to load in the dataset into a tf.data.Dataset object
    train_ds = tfds.load(
        "radio_ml_2018.01", data_dir=dest_path, split="train", as_supervised=True
    )

    # Verify the data was loaded properly
    for data, label in train_ds.take(1):
        print(data.shape)
        print(label)
