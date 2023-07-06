from smart_tfrecord_writer.writer import Writer
from smart_tfrecord_writer.tfrecord_protos import (
    _bytes_feature,
    _float_feature,
    _int64_feature,
)

import tensorflow as tf
import os


class ImageNetWriter(Writer):
    @staticmethod
    def example_definition(filename, image_buffer, label, synset, height, width):
        colorspace = "RGB"
        channels = 3
        image_format = "JPEG"

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "image/height": _int64_feature(height),
                    "image/width": _int64_feature(width),
                    "image/colorspace": _bytes_feature(colorspace),
                    "image/channels": _int64_feature(channels),
                    "image/class/label": _int64_feature(label),
                    "image/class/synset": _bytes_feature(synset),
                    "image/format": _bytes_feature(image_format),
                    "image/filename": _bytes_feature(os.path.basename(filename)),
                    "image/encoded": _bytes_feature(image_buffer),
                }
            )
        )
        return example
