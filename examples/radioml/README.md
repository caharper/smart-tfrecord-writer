# RadioML2018.01A TFRecord Conversion

This directory shows how to convert the RadioML2018.01A dataset to TFRecord format.  The dataset is available for download from [DeepSig](https://www.deepsig.ai/datasets) (**~21.45GB in .hdf5 format**).

**NOTE: your system must have enough disk space to store the .hdf5 and TFRecord dataset. (~43.5GB total).**  Once the TFRecord dataset is created, the .hdf5 dataset can be deleted.

## Setup

1. First, download the dataset from DeepSig.  
2. Adjust the `radioml_to_tfrecord.py` script to point to the downloaded dataset and update the desired destination path.  Specifically, change the `src_path` variable to point to the downloaded dataset.  The `dest_path` should be adjusted to where you want to save your data in TFRecord format.  These are marked with `TODO:` in the script.
3. To convert the dataset to TFRecord format, run the following command:

    ```bash
    python radioml_to_tfrecord.py
    ```

The TFRecord dataset should be **~21.84GB**.  The .hdf5 dataset can be deleted once the TFRecord dataset is created.