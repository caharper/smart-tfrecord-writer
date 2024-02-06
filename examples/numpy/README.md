# NumPy Arrays TFRecord Conversion

This directory shows how to convert numpy arrays to TFRecord format. This is intended as a minimal use case for the average user who has already converted their dataset to numpy arrays and wants to convert them to TFRecord format.

## Setup

1. First, download the dataset from DeepSig.
2. Adjust the `numpy_to_tfrecord.py` script to point to the desired destination path. Specifically, `dest_path` should be adjusted to where you want to save your data in TFRecord format. These are marked with `TODO:` in the script.
3. To convert the dataset to TFRecord format, run the following command:

   ```bash
   python numpy_to_tfrecord.py
   ```

The TFRecord dataset should be **~308KB**. Again, this is a simplistic example and is just provided for demonstration purposes.
