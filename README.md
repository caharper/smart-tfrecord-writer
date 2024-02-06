# smart-tfrecord-writer

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
![Development Status](https://img.shields.io/badge/development-active-brightgreen.svg)
![PyPI Downloads](https://img.shields.io/pypi/dm/smart-tfrecord-writer.svg)

smart-tfrecord-writer helps researchers and practitioners convert their data over to TFRecord format with ease and only a few lines of code! This repo is under active development so please let us know if you encounter bugs, have feature requests, or documentation/code is unclear!

## Table of Contents

- [smart-tfrecord-writer](#smart-tfrecord-writer)
  - [Table of Contents](#table-of-contents)
  - [Why use smart-tfrecord-writer?](#why-use-smart-tfrecord-writer)
  - [Installation](#installation)
    - [PyPI](#pypi)
  - [Usage](#usage)
    - [Required Subclassing Functions](#required-subclassing-functions)
    - [Optional Functions](#optional-functions)
    - [Datastructures for Writer.write_records()](#datastructures-for-writerwrite_records)
    - [Additional Writer.write_records() Parameters](#additional-writerwrite_records-parameters)
  - [Contributing](#contributing)
  - [Why use smart-tfrecord-writer instead of TFDS CLI?](#why-use-smart-tfrecord-writer-instead-of-tfds-cli)
  - [Gotchas](#gotchas)
  - [License](#license)
  - [Contact](#contact)

## Why use smart-tfrecord-writer?

TFRecord format has many benefits over other file formats, but it can be difficult and cumbersome to convert data over to TFRecord format without experience. smart-tfrecord-writer aims to provide a simple, lightweight codebase (**only need to modify 2 small functions!**) to speed up this data transformation and allow researchers and practitioners to focus creating models and less time spent on data conversion. Using smart-tfrecord-writer allows you to utilize great simplifications that come with [Tensorflow Datasets](https://www.tensorflow.org/datasets) like `tfds.load()` and `tfds.features` for your own custom (or not yet TFRecord converted) datasets!

Many of the benefits of using TFRecord format are efficient data storage,
optimized data loading, seamless integration with ML pipelines, etc. More can be learned on the [TensorFlow Documentation](https://www.tensorflow.org/tutorials/load_data/tfrecord).

## Installation

### PyPI

```bash
pip install smart-tfrecord-writer
```

## Usage

smart-tfrecord-writer aims to assist researchers in saving their data into [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) format with miminal code. A base writer class `smart_tfrecord_writer.Writer()` is provided and two (2) **small** functions must be defined when subclassing `Writer`.

### Required Subclassing Functions

1. `Writer.features()` uses the [Tensorflow-Datasets](https://www.tensorflow.org/datasets) (tfds) [features module](https://www.tensorflow.org/datasets/api_docs/python/tfds/features) so that the writer can understand the structure of the data being saved. A simple example can be seen below and in the [examples directory](./examples/radioml/radioml_to_tfrecord.py):

   ```python
   def features(self):
       features = tfds.features.FeaturesDict(
           {
               "rf_signal": tfds.features.Tensor(
                   shape=(1024, 2),
                   dtype=np.float32,
                   doc="A radio signal with I and Q components.",
               ),
               "label": tfds.features.ClassLabel(
                   names=["OOK", "4ASK",..., "OQPSK"],
               ),
               "snr": tfds.features.Scalar(
                   dtype=np.float32, doc="Average SNR of the signal."
               ),
           }
       )

       return features
   ```

   In this example, there are three (3) fields in each example within the dataset: rf_signal, label, and snr. To get a better understanding of the different types of features, we recommend starting at the tfds.features [documentation](https://www.tensorflow.org/datasets/api_docs/python/tfds/features). In general, if your data does not fit within the provided features, `tfds.features.Tensor` is generic enough to provide flexibility.

2. `Writer.process_data()` processes a single element/example within your dataset from the iterator you provide to `Writer.write_records()`. If your data is already processed and is just being iterated over (_e.g._, numpy arrays), then this function can just return the values being iterated over. However, **the output of this function must be a Python dictionary with the same keys as in the features object from `Writer.features()`**. Here is a simple example that can also be found in the [examples directory](./examples/radioml/radioml_to_tfrecord.py):

   ```python
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
   ```

   Notice how `parsed_instance` has the same keys as `features` (rf_signal, label, snr). The values in `parsed_instance` also have datatypes that are supported by the respective components inside `features`.

   If your data pipeline requires more processing, loading images for example, you can adjust the `process_data()` function as needed. It will always receive a single element from the iterator you provide. In the example above, indicies within an hdf5 file were provided. This also could have been filepaths to load, a single row within a numpy array, etc. The main takeaway is how you process a single example.

### Optional Functions

If you would like to provide additional meta information for your dataset, you can also subclass the `Writer.extend_meta_data()` function. This is a simple function that allows you to provide additional details about your dataset. Currently, the supported additional meta information fields are: `description`, `homepage`, `supervised_keys`, and `citation`. `Writer.extend_meta_data()` assumes that these are returned in a tuple with this exact order as seen in the [examples directory](./examples/radioml/radioml_to_tfrecord.py). Some of these are self-explanatory, but a potentially useful field is `supervised_keys`.

If your dataset contains labeled information for classification, you can provide the `(data, label)` pair from your `features` dictionary and use `tfds.load(..., as_supervised=True)` resulting in a [TensorFlow dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) (`tf.data.Dataset`) object that can be iterated over where each iteration yields a `(data, label)` pair that is compatible with most classification models and data pipelines. An example of this can be seen in the [examples directory](./examples/radioml/radioml_to_tfrecord.py).

`extended_dataset_info()` may also be subclassed and will write additional dataset information to `extended_dataset_info.json`. This may be helpful if you have additional information about your dataset that does not conform the the `tfds.DatasetInfo` object. This is a simple function that returns a dictionary of the information you would like to save. Because `extended_dataset_info()` is saved into a json file, the values must be json serializable. An example can be seen below:

```python
        def extended_dataset_info(self):
            return {
                "additional_info": "This is an example of additional dataset information.",
                "more_info": "This is another example of additional dataset information.",
            }
```

### Datastructures for Writer.write_records()

If using `splits_info`, the assumed data structure is a list of Python dictionaries. Each list element is assumed to contain a two key value pairs with the structure:

```python
{"name": "<split_name>", "info": split_info}
```

Replace `<split_name>` with a string of the name of the respective split (_e.g._, _train_, _val_, _test_, _etc_.). For example:

```python
    # Example indices
    train_indexes = list(range(3_000))
    test_indexes = list(range(1_000))

    # Structure splits information for the writer into a list of dictionaries
    splits_info = [
        {"name": "train", "info": train_indexes},
        {"name": "test", "info": test_indexes},
    ]
```

`split_info` is intended to be generic to fit many use cases. The only restriction is that it needs to be an iterable where a single element can be processed by the `Writer.process_data()` function. For example, this could be a list of indices, file names, etc.! The only limitation currently is that the length of the iterator (`len(iter)`) must be defined. This is a result of wanting to make our code compatible with the `tfds.load()` functionality. Support for unknown length iterators may be added in the future. An example usage of this can be seen in the [examples directory](./examples/radioml/radioml_to_tfrecord.py) where the iterator is a list of indices for train and test splits.

If you want to provide you own dataset splits, you can pass `splits_shards` instead of `split_info` to `Writer.write_records()` with a similar datastructure, but instead of `"info"`, the key will be `"shards"` and each shards element is an iterable. For example:

```python
      # Structure shards information for the writer into a list of dictionaries
      splits_info = [
          {"name": "train", "shards": [[1, 2, 3], [4, 5, 6]]},
          {"name": "test", "shards": [[7, 8, 9], [10, 11, 12]]},
      ]
```

### Additional Writer.write_records() Parameters

- `shuffle`: Whether or not to shuffle the provided data in `"info"` or `"shards`".
- `random_seed`: Random seed used for shuffling for reproducibility
- `examples_per_shard`: If you know how many examples you want in each shard, provide it here!
- `mb_per_shard`: If you aren't sure how many examples you want per shard, but you know the approximate amount of memory for each shard you want to use, allow smart-tfrecord-writer to figure out how many examples per shard for you! A good rule of thumb is 100 MB+ to take advantage of parallelism. More on that in [TensorFlow documentation](https://www.tensorflow.org/tutorials/load_data/tfrecord). Note that this is just an estimate and does not guarantee an exact amount of memory per shard.
- `n_estimates_mb_per_example`: If you are using `mb_per_shard`, `n_estimates_mb_per_example`: Tells smart-tfrecord-writer how many examples it should iterate over to get an estimate for how much memory each example will use. If your data is uniform in shape (_e.g._, all images are $224 \times 224 \times 3$), then 1 is sufficient. If your data is variable shaped, the higher this number is, the better the estimate, but will take more time the higher the value is set.
- `n_estimates_mb_per_split_example`: Similar to `n_estimates_mb_per_example`, but allows a dictionary of `{split: num_estimates}` to be used so that each split computes a different estimate for examples per shard. May be useful if the evaluation set has differing shapes than the training set for example.

## Contributing

Contributions are welcome! If you'd like to contribute to the project, please follow these guidelines:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Test your changes thoroughly.
5. Submit a pull request.
6. Add caharper as a reviewer.

Please ensure your code adheres to the project's coding style and conventions.

## Why use smart-tfrecord-writer instead of TFDS CLI?

The TFDS CLI is great! The purpose of smart-tfrecord-writer is to further reduce the complexity of formatting data and saving data into the TFRecord format. Instead of having to read verbose documentation, we hope to make the process as simple as possible by providing just a few parameters and overwriting a few functions to get the most out of your ML pipeline.

Most projects could benefit from using TFRecords, but it requires a lot of reading, looking at several coding examples, etc. Our project aims to help limit the time spent on converting your data over to the TFRecord format so you can take advantage of the speed of TFRecord and start training your models and not waste time elsewhere. TFDS CLI will give you more control at the cost of more documentation and formatting, but most datasets can be handled in a simpler way, enter smart-tfrecord-writer!

## Gotchas

Right now, `tf.data.Dataset` objects are not supported. A few assumptions in our codebase (knowing the length of the dataset for splitting into shards and writing individual shards instead of looping over a single object) cause this limitation. The second is problematic for a `tf.data.Dataset` because it is not guaranteed to be deterministic so we would need to have a separate handler for `tf.data.Dataset` objects.

## License

This project is licensed under the [Apache License](LICENSE).

## Contact

If you have any questions, suggestions, or feedback, feel free to reach out:

- Email: [caharper@smu.edu](mailto:caharper@smu.edu)
- GitHub: [caharper](https://github.com/caharper)
