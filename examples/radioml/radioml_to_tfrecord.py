import tensorflow as tf
from smart_tfrecord_writer.writer import Writer
from smart_tfrecord_writer.tfrecord_protos import (
    _bytes_feature,
    _float_feature,
    _int64_feature,
)
import h5py
import numpy as np


class RadioMLWriter(Writer):
    def example_definition(self, instance):
        observation = instance["observation"]
        label = instance["label"]
        snr = instance["snr"]

        data_dict = {
            "observation": _float_feature(observation.flatten().tolist()),
            "label": _int64_feature(list(label)),
            "snr": _int64_feature(list(snr)),
        }

        # create an Example
        out = tf.train.Example(features=tf.train.Features(feature=data_dict))

        return out.SerializeToString()

    def process_data(self, index):
        parsed_instance = {}

        with h5py.File(self.source_directory, "r") as f:
            observation = f["X"][index]
            label = f["Y"][index]
            snr = f["Z"][index]

            parsed_instance["observation"] = observation
            parsed_instance["label"] = label
            parsed_instance["snr"] = snr

        return parsed_instance


if __name__ == "__main__":
    src_path = "./../../../Modulation/dataset/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5"
    dest_path = "./dest"
    writer = RadioMLWriter(
        source_directory=src_path,
        destination_directory=dest_path,
    )

    # train_indexes = np.genfromtxt("train_indexes.csv", delimiter=",", dtype=np.int)
    # test_indexes = np.genfromtxt("test_indexes.csv", delimiter=",", dtype=np.int)
    train_indexes = list(range(31_653))
    test_indexes = list(range(31_653))
    splits_info = [
        {"name": "train", "info": train_indexes},
        # {"name": "test", "info": test_indexes},
    ]

    writer.write_records(
        splits_info=splits_info,
        mb_per_shard=250,
        n_estimates_mb_per_example=1,
        shuffle=True,
        verbose=3,
    )
