import tensorflow as tf
import tensorflow_datasets as tfds
import math
import warnings
from sklearn.utils import shuffle as sklearn_shuffle
from tqdm import tqdm
import os
import termcolor
import json
import multiprocessing
from functools import partial
from smart_tfrecord_writer.utils import convert_to_human_readable


class Writer(object):
    def __init__(
        self,
        destination_directory,
        config=None,
        version="1.0.0",
    ):
        self.destination_directory = destination_directory

        if not os.path.exists(self.destination_directory):
            os.mkdir(self.destination_directory)

        self.version = version

        if config is None:
            config = self.destination_directory.split("/")[-1]
        self.config = config

        if not os.path.exists(os.path.join(self.destination_directory, self.config)):
            os.mkdir(os.path.join(self.destination_directory, self.config))

        if not os.path.exists(
            os.path.join(self.destination_directory, self.config, self.version)
        ):
            os.mkdir(
                os.path.join(self.destination_directory, self.config, self.version)
            )

        self._validated = False

    def extended_dataset_info(self):
        return None

    @property
    def _extended_dataset_info(self):
        extended_info = self.extended_dataset_info()
        if extended_info is not None:
            # Verify extended data is json serializable
            try:
                json.dumps(extended_info)
            except TypeError as e:
                print(f"Extended dataset info is not json serializable.  Error: {e}")

        return extended_info

    def _write_extended_dataset_info(self):
        if self._extended_dataset_info is not None:
            extended_dataset_info_path = os.path.join(
                self.destination_directory,
                self.config,
                self.version,
                "extended_dataset_info.json",
            )
            # Write extended dataset info to json
            with open(extended_dataset_info_path, "w") as f:
                json.dump(self._extended_dataset_info, f, indent=4)

    def extend_meta_data(self):
        description = ""
        homepage = ""
        supervised_keys = None
        citation = ""

        return description, homepage, supervised_keys, citation

    def _write_meta_data(self, split_infos):
        (
            description,
            homepage,
            supervised_keys,
            citation,
        ) = self.extend_meta_data()

        tfds.folder_dataset.write_metadata(
            data_dir=os.path.join(
                self.destination_directory, self.config, self.version
            ),
            features=self._features,
            split_infos=split_infos,
            filename_template="{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}",
            description=description,
            homepage=homepage,
            version=self.version,
            supervised_keys=supervised_keys,
            citation=citation,
        )

    def features(self):
        """Define entries within a single TFRecord example proto

        Assumes the output will be serialized with .SerializeToString()
        """
        raise NotImplementedError(f"features must be defined when subclassing Writer")

    @property
    def _features(self):
        return self.features()

    def process_data(self, example):
        """Processing to clean the data before converting the data over to a TFRecord
        example.  For instance, process_data(filename) would accept a file name, load
        the data from the file, perform requisite processing, and return the data in the
        desired saving format.

        If train_info, val_info, or test_info are passed to `write_records`,
        `process_data` will operate under a single example from the info passed to
        `write_records`.

        Must return a named dictionary (so that _estimate_mb_per_example will work)
        """
        raise NotImplementedError(
            f"process_data must be defined when subclassing Writer"
        )

    def _process_data(self, example):
        processed_data = self.process_data(example)

        # Validate data if not already validated
        if not self._validated:
            self._validate_data(processed_data)

        return self._features.serialize_example(processed_data)

    def _validate_data(self, example):
        feature_keys = list(self._features.keys())
        for key in list(example.keys()):
            if key not in self._features:
                raise ValueError(
                    f"{key} <from `process_data()` is not a valid feature "
                    f"for the dataset.  Choose from the keys created in "
                    f"`features()`.  Valid keys defined in `features()`: {feature_keys}"
                )

        self._validated = True

    def _estimate_bytes_per_example(self, info, n_estimates, verbose=0):
        if verbose > 0:
            print("Estimating bytes per example...")

        total_bytes = 0

        for instance in info[:n_estimates]:
            # Serialize the data
            serialized_data = self._process_data(instance)
            total_bytes += len(serialized_data)

        avg_bytes_per_example = total_bytes / n_estimates
        return avg_bytes_per_example

    def _estimate_n_examples_per_shard(
        self, mb_per_shard, info, n_estimates, verbose=0
    ):
        avg_bytes_per_example = self._estimate_bytes_per_example(
            info, n_estimates, verbose=verbose
        )
        byte_per_shard = mb_per_shard * (1024**2)
        n_examples_per_shard = math.ceil((1 / avg_bytes_per_example) * byte_per_shard)
        return n_examples_per_shard

    def _write_shard(self, shard_info_and_name, verbose=0):
        shard_info, shard_name = shard_info_and_name

        # Writes a single shard to a file
        if verbose >= 3:
            shard_pbar = tqdm(
                shard_info, desc=f"Writing shard {shard_name}", leave=False
            )
        else:
            shard_pbar = shard_info

        shard_name = os.path.join(
            self.destination_directory, self.config, self.version, shard_name
        )
        with tf.io.TFRecordWriter(shard_name) as out:
            for example in shard_pbar:
                # Serialize the data
                serialized_data = self._process_data(example)
                out.write(serialized_data)

        return os.path.getsize(shard_name)

    def _write_shards(self, splits_shards, verbose=0):
        if verbose > 0:
            split_pbar = tqdm(splits_shards, desc="Writing splits")  # 1, ...,
        else:
            split_pbar = splits_shards

        split_infos = []

        for split in split_pbar:
            split_name = split["name"]
            num_shards = len(split["shards"])
            dest_base = self.destination_directory.split("/")[-1]
            shard_names = [
                f"{dest_base}-{split_name}.tfrecord-{i:05d}-of-{num_shards:05d}"
                for i in range(len(split["shards"]))
            ]

            if verbose >= 2:
                shard_pbar = tqdm(
                    zip(split["shards"], shard_names),
                    desc=f"Writing {split_name} shards",
                    total=len(split["shards"]),
                    leave=False,
                )
            else:
                shard_pbar = zip(split["shards"], shard_names)

            split_bytes = 0

            for shard_info, shard_name in shard_pbar:

                # Write single shard
                bytes = self._write_shard((shard_info, shard_name), verbose=verbose)

                # Get bytes information
                split_bytes += bytes

            # Create split information
            split_infos.append(
                tfds.core.SplitInfo(
                    name=split_name,
                    shard_lengths=[len(shard) for shard in split["shards"]],
                    num_bytes=split_bytes,
                )
            )

        self._write_meta_data(split_infos)

    @staticmethod
    def split(a, n):
        """Splits a into n approximately equal parts.

        Credit to user tixxit: https://stackoverflow.com/a/2135920

        Parameters
        ----------
        a : array-like
            Array to be split
        n : int
            Number of splits

        Returns
        -------
        array-like
            n sub-arrays of a of approximately equal size
        """
        k, m = divmod(len(a), n)
        return [a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]

    def _create_shards(self, info, examples_per_shard):
        n_shards = math.ceil(len(info) / examples_per_shard)

        # Split into shards
        split_shards = Writer.split(info, n_shards)

        return split_shards

    def write_records(
        self,
        splits_info=None,
        splits_shards=None,
        shuffle=True,
        random_seed=42,
        examples_per_shard=None,
        mb_per_shard=None,
        n_estimates_mb_per_example=1,
        n_estimates_mb_per_split_example=None,
        verbose=0,
    ):
        """Writes data into a TFRecord format.

        Parameters
        ----------
        splits_info : list, optional
            List of dictionaries containing information to be iterated over, by default
            None
        splits_shards : list, optional
            List of precomputed shards to be saved and iterated over, by default None
        shuffle : bool, optional
            Whether or not to shuffle the data provided, by default True
        random_seed : int, optional
            Random seed provided to shuffle for reproducibility, by default 42
        examples_per_shard : int, optional
            Number of examples per shard, by default None
        mb_per_shard : number, optional
            Allow writer to estimate the number of examples per shard with a target
            memory usage in MB for each shard, by default None
        n_estimates_mb_per_example : int, optional
            Number of estimates to compute the number of examples per shard if
            mb_per_shard is provided, by default 1
        n_estimates_mb_per_split_example : dict, optional
            Similar to n_estimates_mb_per_example but for each individual split, by
            default None
        verbose : int, optional
            How much logging information to display.  Values are 0 (silent), 1, 2, 3
            (most), by default 0
        """

        # Validate parameters
        if splits_info is None and splits_shards is None:
            raise ValueError(
                f"Information or shards must be supplied for at least one split."
            )

        if splits_info is not None and splits_shards is None:
            contains_info, split_names = zip(
                *[(split["info"] is not None, split["name"]) for split in splits_info]
            )

            if not any(contains_info):
                raise ValueError(
                    f"Information must be supplied for at least one of "
                    f"{' ,'.join(split_names)}."
                )

        if splits_info is None and splits_shards is not None:
            contains_shards, split_names = zip(
                *[(split["shards"] is not None, split["name"]) for split in splits_info]
            )

            if not any(contains_shards):
                raise ValueError(
                    f"Information must be supplied for at least one of "
                    f"{' ,'.join(split_names)}."
                )

            if mb_per_shard is not None:
                raise ValueError(f"If shards are supplied, mb_per_shard must be None.")

        if mb_per_shard is not None:
            if mb_per_shard <= 0:
                raise ValueError(f"mb_per_shard must be > 0.")

            if n_estimates_mb_per_split_example is not None:
                if n_estimates_mb_per_example is not None:
                    warnings.warn(
                        "Ignoring n_estimates_mb_per_example as "
                        "n_estimates_mb_per_split_example is not None.  This may not "
                        "be desired behavior."
                    )

        # Setup shards
        if splits_shards is not None:
            if shuffle:
                for split_shard in splits_shards:
                    split_shard["shards"] = sklearn_shuffle(
                        split_shard["shards"], random_state=random_seed
                    )

        # Setup shards
        if splits_info is not None:
            splits_shards = []
            for split in splits_info:
                split_shard_info = {}
                split_name = split["name"]
                split_info = split["info"]

                if shuffle:
                    split_info = sklearn_shuffle(split_info, random_state=random_seed)

                if examples_per_shard is not None:
                    split_examples_per_shard = examples_per_shard

                else:
                    split_examples_per_shard = split.get("examples_per_shard")

                if split_examples_per_shard is None and mb_per_shard is None:
                    raise ValueError(
                        f"Either mb_per_shard must be specified or examples_per_shard "
                        f"must be specified in splits_info for {split_name} split."
                    )
                if split_examples_per_shard is not None and mb_per_shard is not None:
                    warnings.warn(
                        f"Ignoring mb_per_shard as split_examples_per_shard is not "
                        f"None for {split_name} split.  This may not be desired "
                        "behavior."
                    )

                # If compute elements per shard based on memory
                if split_examples_per_shard is None and mb_per_shard is not None:
                    # Estimate for each split
                    n_estimates = None
                    if n_estimates_mb_per_split_example is not None:
                        n_estimates = n_estimates_mb_per_split_example.get(split_name)
                        if (
                            n_estimates is not None
                            and n_estimates_mb_per_example is not None
                        ):
                            warnings.warn(
                                f"Ignoring n_estimates_mb_per_example as "
                                f"n_estimates_mb_per_split_example is not None for "
                                f"{split_name} split.  This may not be desired "
                                f"behavior."
                            )
                    else:
                        n_estimates = n_estimates_mb_per_example

                    validate_n_estimates(n_estimates, f"  For {split_name} split.")

                    split_examples_per_shard = self._estimate_n_examples_per_shard(
                        mb_per_shard,
                        split_info,
                        n_estimates_mb_per_example,
                        verbose=verbose,
                    )

                print(
                    f"computed split_examples_per_shard: {split_examples_per_shard} "
                    f"for {split_name} split."
                )

                # Split the info into shards
                split_shard_info["shards"] = self._create_shards(
                    split_info, split_examples_per_shard
                )

                if verbose > 0:
                    n_shards = len(split_shard_info["shards"])

                    # Convert MB to human readable form in GB/MB
                    total_mb = n_shards * mb_per_shard
                    termcolor.cprint(
                        termcolor.colored(
                            f"Computed {n_shards} shards for {split_name} split. "
                            f"Estimate: {convert_to_human_readable(total_mb)}",
                            "green",
                            attrs=["bold"],
                        )
                    )

                # Populate splits_shards so they can be written
                split_shard_info["name"] = split_name

                splits_shards.append(split_shard_info)

        # Write shards
        self._write_shards(splits_shards, verbose=verbose)

        # Write extended dataset info
        self._write_extended_dataset_info()


def validate_n_estimates(n_estimates, err=""):
    if not isinstance(n_estimates, int):
        err = "Number of estimates must be an integer." + err
        raise ValueError(err)


def check_parameters(param1, param2):
    if (param1 is not None and param2 is None) or (
        param1 is None and param2 is not None
    ):
        # One parameter is not None and the other is None
        return False
    return True
