# from abc import abstractmethod
import tensorflow as tf
import math
import warnings
import numpy as np
from sklearn.utils import shuffle as sklearn_shuffle
from tqdm import tqdm
import os


class Writer(object):
    def __init__(self, source_directory, destination_directory):
        self.source_directory = source_directory
        self.destination_directory = destination_directory

        if not os.path.exists(source_directory):
            raise ValueError(f"Source directory, {source_directory}, does not exist.")

        if not os.path.exists(destination_directory):
            os.mkdir(destination_directory)

    # @staticmethod
    def example_definition(self, instance):
        """Define entries within a single TFRecord example proto

        Assumes the output will be serialized with .SerializeToString()
        """
        raise NotImplementedError(
            f"example_definition must be defined when subclassing Writer"
        )

    # @staticmethod
    def process_data(self, example):
        """Processing to clean the data before converting the data over to a TFRecord
        example.  For instance, process_data(filename) would accept a file name, load
        the data from the file, perform requisite processing, and return the data in the
        desired saving format.

        If train_info, val_info, or test_info are passed to `write_records`,
        `process_data` will operate under a single example from the info passed to
        `write_records`.

        ***Must return a named dictionary (so that _estimate_mb_per_example will work)
        """
        raise NotImplementedError(
            f"process_data must be defined when subclassing Writer"
        )

    def _estimate_bytes_per_example(self, info, n_estimates, verbose=0):
        if verbose > 0:
            print("Estimating bytes per example...")

        total_bytes = 0

        for instance in info[:n_estimates]:
            processed_data = self.process_data(instance)
            serialized_data = self.example_definition(processed_data)
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

    def _write_shard(self, shard_info, shard_name, verbose=0):
        # Writes a single shard to a file
        if verbose >= 3:
            shard_pbar = tqdm(
                shard_info, desc=f"Writing shard {shard_name}", leave=False
            )
        else:
            shard_pbar = shard_info

        with tf.io.TFRecordWriter(
            os.path.join(self.destination_directory, shard_name)
        ) as out:
            for example in shard_pbar:
                processed_data = self.process_data(example)
                serialized_data = self.example_definition(processed_data)
                out.write(serialized_data)

    def _write_shards(self, splits_shards, verbose=0):
        if verbose > 0:
            split_pbar = tqdm(splits_shards, desc="Writing splits")  # 1, ...,
        else:
            split_pbar = splits_shards

        for split in split_pbar:
            split_name = split["name"]
            num_shards = len(split["shards"])
            if verbose >= 2:
                shard_pbar = tqdm(
                    split["shards"],
                    desc=f"Writing {split_name} shards",
                    leave=False,
                )
            else:
                shard_pbar = split["shards"]

            for i, shard in enumerate(shard_pbar, 1):
                # Get the shard name
                shard_name = f"{split_name}-{i}-of-{num_shards}.tfrecord"

                # Write single shard
                self._write_shard(shard, shard_name, verbose=verbose)

    def _create_shards(self, info, examples_per_shard):
        n_shards = math.ceil(len(info) / examples_per_shard)

        # Split into shards
        split_shards = np.array_split(info, n_shards)

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
        """_summary_

        Parameters
        ----------
        train_info : array-like, optional
            Information for training data.  Each entry will be iterated over as a single
            instance processed by `process_data`.  Cannot be supplied if `train_shards`
            is also supplied.  By default None.
        train_shards : _type_, optional
            _description_, by default None
        val_info : _type_, optional
            _description_, by default None
        val_shards : _type_, optional
            _description_, by default None
        test_info : _type_, optional
            _description_, by default None
        test_shards : _type_, optional
            _description_, by default None
        shuffle : bool, optional
            _description_, by default True
        random_seed : int, optional
            _description_, by default 42
        files_per_shard : _type_, optional
            _description_, by default None
        mb_per_shard : _type_, optional
            _description_, by default None
        n_estimates_mb_per_example : int, optional
            Number of examples to iterate over to estimate the amount of megabytes per
            example.  This only runs if mb_per_shard is not None.  If fixed length
            features are the only components to examples, then 1 should be sufficient as
            all other examples will have the same memory usage and is therefore a
            perfect estimate.  If variable length features are in examples, then higher
            values of `n_estimates_mb_per_example` will lead to more accurate estimates
            at the expense of more computational overhead. By default None.
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
                    f"Information must be supplied for at least one of {' ,'.join(split_names)}."
                )

        if splits_info is None and splits_shards is not None:
            contains_shards, split_names = zip(
                *[(split["shards"] is not None, split["name"]) for split in splits_info]
            )

            if not any(contains_shards):
                raise ValueError(
                    f"Information must be supplied for at least one of {' ,'.join(split_names)}."
                )

            if mb_per_shard is not None:
                raise ValueError(f"If shards are supplied, mb_per_shard must be None.")

        if mb_per_shard is not None:
            if mb_per_shard <= 0:
                raise ValueError(f"mb_per_shard must be > 0.")

            if n_estimates_mb_per_split_example is not None:
                if n_estimates_mb_per_example is not None:
                    warnings.warn(
                        "Ignoring n_estimates_mb_per_example as n_estimates_mb_per_split_example is not None.  This may not be desired behavior."
                    )

                ...

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
                        f"Either mb_per_shard must be specified or examples_per_shard must be specified in splits_info for {split_name} split."
                    )
                if split_examples_per_shard is not None and mb_per_shard is not None:
                    warnings.warn(
                        f"Ignoring mb_per_shard as split_examples_per_shard is not None for {split_name} split.  This may not be desired behavior."
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
                                f"Ignoring n_estimates_mb_per_example as n_estimates_mb_per_split_example is not None for {split_name} split.  This may not be desired behavior."
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

                # Instead of splitting into shards, it may be better to just compute the
                # number of elements in each shard.  This would make the code more
                # compatible with tf.data.Datasets.

                # Split the info into shards
                split_shard_info["shards"] = self._create_shards(
                    split_info, split_examples_per_shard
                )

                if verbose > 0:
                    n_shards = len(split_shard_info["shards"])
                    print("Computed {n_shards} shards for {split_name} split.")

                # Populate splits_shards so they can be written
                split_shard_info["name"] = split_name

                splits_shards.append(split_shard_info)

        # Get split meta information

        # Write shards
        self._write_shards(splits_shards, verbose=verbose)

    def _get_split_meta_info(self, split):
        shards = split["shards"]


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
