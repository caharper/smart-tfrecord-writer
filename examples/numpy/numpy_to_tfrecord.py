import tensorflow_datasets as tfds
from smart_tfrecord_writer import Writer
import numpy as np


class NumpyWriter(Writer):
    def extend_meta_data(self):
        """Optional method that allows additional information to be stored in meta data.

        Returns
        -------
        tuple
            Components for the description, homepage, supervised keys, and citation.
        """
        description = """
        A simple dataset used for illustrative purposes created from randomly generated
        numpy arrays. This dataset is used to demonstrate how to use the 
        smart_tfrecord_writer on already processed/mostly processed data.
        """

        homepage = ""
        supervised_keys = ("data", "label")
        citation = ""

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
                "data": tfds.features.Tensor(
                    shape=(100, 2),
                    dtype=np.float32,
                    doc="A random numpy array.",
                ),
                "label": tfds.features.ClassLabel(
                    names=[
                        "True",
                        "False",
                    ],
                    doc="A random binary label.",
                ),
            }
        )

        return features

    def process_data(self, example):
        """Required function that processes the data for a single example

        Parameters
        ----------
        example : tuple of (np.ndarray, int)
            Data and label for a single example.

        Returns
        -------
        dict
            Parsed instance of the example.
        """
        data, label = example
        parsed_instance = {
            "data": data,
            "label": np.squeeze(label),  # ClassLabel expects a scalar
        }

        return parsed_instance


if __name__ == "__main__":
    # TODO: Add paths to the destination files
    dest_path = "/path/to/numpy_tfrecord"
    writer = NumpyWriter(
        destination_directory=dest_path,
    )

    # Create random numpy arrays of shape (100, 2)
    train_data = np.random.rand(200, 100, 2).astype(np.float32)
    validation_data = np.random.rand(100, 100, 2).astype(np.float32)
    test_data = np.random.rand(50, 100, 2).astype(np.float32)

    # Create random labels for the data
    train_labels = np.random.randint(0, 2, 200)
    validation_labels = np.random.randint(0, 2, 100)
    test_labels = np.random.randint(0, 2, 50)

    # Structure splits information for the writer into a list of dictionaries
    splits_info = [
        {"name": "train", "info": list(zip(train_data, train_labels))},
        {
            "name": "validation",
            "info": list(zip(validation_data, validation_labels)),
        },
        {"name": "test", "info": list(zip(test_data, test_labels))},
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
        "numpy_tfrecord", data_dir=dest_path, split="train", as_supervised=True
    )

    # Verify the data was loaded properly
    for data, label in train_ds.take(1):
        print(data.shape)  # => (100, 2)
        print(label)  # => 0 or 1
