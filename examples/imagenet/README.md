# ImageNet TFRecord Conversion

This directory shows how to convert the ImageNet dataset to TFRecord format.  For simplicity, we do not include the annotations with bounding boxes, just the classification labels as is common for most image classification tasks using ImageNet.  The dataset is available for download from [Kaggle](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data) (**~166.5GB in .zip form**).

The conversion script makes use of temporary files and reads directly from the zip file so that extra memory is not required.  This may come at some cost to speed, but it is a tradeoff we are willing to make for memory complexity.

**NOTE: your system must have enough disk space to store the zip and TFRecord dataset. (~674.5GB total).**

## Steps

1. Download the dataset from Kaggle along with the accompanying files (csv and txt files).
   `NOTE:` There are some (6 total) corrupted and missing files in the Kaggle zip file.  Namely:

   ```python
   [
        "ILSVRC/Data/CLS-LOC/train/n01692333/n01692333_9111.JPEG",
        "ILSVRC/Data/CLS-LOC/train/n01981276/n01981276_8715.JPEG",
        "ILSVRC/Data/CLS-LOC/train/n02206856/n02206856_18.JPEG",
        "ILSVRC/Data/CLS-LOC/train/n02510455/n02510455_9459.JPEG",
        "ILSVRC/Data/CLS-LOC/train/n02951585/n02951585_18456.JPEG",
        "ILSVRC/Data/CLS-LOC/train/n03337140/n03337140_6054.JPEG",
    ]
    ```

   This script will automatically skip these files.

2. Search for `TODO:` in `imagenet_to_tfrecord.py` and update the paths to point to the downloaded dataset and update the desired destination path.  Specifically, change the `src_path` variable to point to the downloaded dataset.  The `dest_path` should be adjusted to where you want to save your data in TFRecord format.  You will also need to point to the csv files for the validation set.
3. Run the following command to convert the dataset to TFRecord format:

    ```bash
    python imagenet_to_tfrecord.py
    ```

    And you're done!  The TFRecord dataset should be **~508GB**.  The .zip dataset can be deleted once the TFRecord dataset is created (or you can keep it around if you want to use it for other purposes such as object detection, etc.).
