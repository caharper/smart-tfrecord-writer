import tensorflow_datasets as tfds
from smart_tfrecord_writer import Writer
import numpy as np
from PIL import Image
import numpy as np
import os
import zipfile
import tempfile
import pandas as pd


# Wordnet IDs for the 1000 classes in ImageNet
# TODO: Add path to the synset labels file
synset_labels_path = "path/to/LOC_synset_mapping.txt"
wnids_labels = []
with open(synset_labels_path, "r") as f:
    for line in f:
        wnids_labels.append(line.split()[0])

# Add unknown for test set (1001st class)
wnids_labels.append("unknown")

assert len(wnids_labels) == 1001

# Create a dictionary of the validation labels
# TODO: Add path to the validation labels file
val_soln_df = pd.read_csv("path/to/LOC_val_solution.csv")
ids = val_soln_df["ImageId"]
labels = [pred_string.split(" ")[0] for pred_string in val_soln_df["PredictionString"]]
val_label_dict = {id_: label for id_, label in zip(ids, labels)}


class ImageNetWriter(Writer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zip_ref = zipfile.ZipFile(self.source_directory, "r")

    def __del__(self):
        self.zip_ref.close()

    def extend_meta_data(self):
        """Optional method that allows additional information to be stored in meta data.

        Returns
        -------
        tuple
            Components for the description, homepage, supervised keys, and citation.
        """
        description = """
        ImageNet is an image database organized according to the WordNet hierarchy 
        (currently only the nouns), in which each node of the hierarchy is depicted by 
        hundreds and thousands of images. The project has been instrumental in advancing 
        computer vision and deep learning research. The data is available for free to 
        researchers for non-commercial use."""

        homepage = "https://www.image-net.org/"
        supervised_keys = ("image", "label")
        citation = """
        @INPROCEEDINGS{5206848,
            author={Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Kai Li and Li Fei-Fei},
            booktitle={2009 IEEE Conference on Computer Vision and Pattern Recognition}, 
            title={ImageNet: A large-scale hierarchical image database}, 
            year={2009},
            volume={},
            number={},
            pages={248-255},
            doi={10.1109/CVPR.2009.5206848}}
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
                "image": tfds.features.Image(
                    shape=(None, None, 3),
                    dtype=np.uint8,  # Image feature expects uint8
                    doc="Image from the ImageNet dataset.",
                ),
                "label": tfds.features.ClassLabel(
                    names=wnids_labels,
                ),
                "file_name": tfds.features.Text(doc="Base file name of the image."),
            }
        )

        return features

    def process_data(self, filepath):
        image = None

        # Create a temporary directory to store the decompressed files
        with tempfile.TemporaryDirectory(dir=self.destination_directory) as temp_dir:
            # Decompress the file
            self.zip_ref.extract(filepath, path=temp_dir)

            tmp_file_path = os.path.join(temp_dir, filepath)

            # Load the image
            image = Image.open(tmp_file_path)

        # Convert grayscale to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Convert the image to a NumPy array
        image_array = np.asarray(image)

        # Verify the image is in uint8 format
        assert image_array.dtype == np.uint8

        # Verify the image has 3 channels
        image_array = image_array[:, :, :3]

        # Get the label of the image
        if "test" in filepath:
            label = "unknown"
        elif "val" in filepath:
            # Get the filepath without the extension
            file_name = os.path.splitext(os.path.basename(filepath))[0]

            # Find the corresponding label in the validation labels dictionary
            label = val_label_dict[file_name]
        else:
            label = filepath.split("/")[-2]

        # Get the base filepath of the image
        file_name = filepath.split(
            os.path.join(self.source_directory, "ILSVRC", "Data", "CLS-LOC")
        )[-1]

        return {"image": image_array, "label": label, "file_name": file_name}


if __name__ == "__main__":
    # Source and destination files
    # TODO: Add paths to the source and destination files
    src_path = "path/to/imagenet-object-localization-challenge.zip"
    dest_path = "/path/to/output/imagenet_tfrecord"
    writer = ImageNetWriter(
        source_directory=src_path,
        destination_directory=dest_path,
    )

    file_list = []
    # Open the zip file
    with zipfile.ZipFile(src_path, "r") as zip_ref:
        # Get the file names in the zip file
        file_list = zip_ref.namelist()

    # Subset the file list to only include the data files (no annotations)
    data_subset_list = [file for file in file_list if "ILSVRC/Data/" in file]

    # Bad CRC-32 checksums for the following files (all training files)
    bad_crc_files = [
        "ILSVRC/Data/CLS-LOC/train/n01692333/n01692333_9111.JPEG",
        "ILSVRC/Data/CLS-LOC/train/n01981276/n01981276_8715.JPEG",
        "ILSVRC/Data/CLS-LOC/train/n02206856/n02206856_18.JPEG",
        "ILSVRC/Data/CLS-LOC/train/n02510455/n02510455_9459.JPEG",
        "ILSVRC/Data/CLS-LOC/train/n02951585/n02951585_18456.JPEG",
        "ILSVRC/Data/CLS-LOC/train/n03337140/n03337140_6054.JPEG",
    ]

    # Train, test, validations files
    train_files = [file for file in data_subset_list if "train" in file]
    val_files = [file for file in data_subset_list if "val" in file]
    test_files = [file for file in data_subset_list if "test" in file]

    # Filter training files
    train_files = [file for file in train_files if file not in bad_crc_files]
    print(f"Number of train files: {len(train_files)}")
    print(f"Number of val files: {len(val_files)}")
    print(f"Number of test files: {len(test_files)}")

    # Structure splits information for the writer into a list of dictionaries
    splits_info = [
        {"name": "train", "info": train_files},
        {"name": "validation", "info": val_files},
        {"name": "test", "info": test_files},
    ]

    # Write the records and have approximately 250MB per shard
    writer.write_records(
        splits_info=splits_info,
        mb_per_shard=250,
        n_estimates_mb_per_example=20,
        shuffle=True,
        verbose=3,
    )

    # Use TFDS to load in the dataset into a tf.data.Dataset object
    train_ds = tfds.load(
        "imagenet_tfrecord",
        data_dir=dest_path,
        split="test",
    )

    # Verify the data was loaded properly
    for data in train_ds.take(1):
        print(data["image"].shape)
        print(data["file_name"])
        print(data["label"])
