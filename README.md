# Project Name

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
![Development Status](https://img.shields.io/badge/development-active-brightgreen.svg)


A brief description of your project.

## Table of Contents

- [Project Name](#project-name)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [Why use smart-tfrecord-writer instead of TFDS CLI?](#why-use-smart-tfrecord-writer-instead-of-tfds-cli)
  - [License](#license)
  - [Contact](#contact)

## Installation

Provide instructions on how to install and set up your project. Be sure to include any dependencies or prerequisites.

## Usage

Explain how to use your project, provide code examples, and demonstrate its features. You can also include screenshots or GIFs to give users a visual understanding.

If using `splits_info`, the assumed data structure is a list of Python dictionaries.  Each list element is assumed to contain a two key value pairs with the structure:

```python
{"name": "<split_name>", "info": split_info}
```

Replace `<split_name>` with a string of the name of the respective split (*e.g.*, *train*, *val*, *test*, *etc*.).

`split_info` is intended to be generic to fit many use cases.  The only restriction is that it needs to be an iterable where a single element can be processed by the `process_data` function.  For example, this could be a list of indices, file names, or even an existing `tf.data.Dataset` object!

## Contributing

Contributions are welcome! If you'd like to contribute to the project, please follow these guidelines:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Test your changes thoroughly.
5. Submit a pull request.

Please ensure your code adheres to the project's coding style and conventions.

## Why use smart-tfrecord-writer instead of TFDS CLI?

The TFDS CLI is great!  The purpose of smart-tfrecord-writer is to further reduce the complexity of formatting data and saving data into the TFRecord format.  Instead of having to read verbose documentation, we hope to make the process as simple as possible by providing just a few parameters and overwriting a few functions to get the most out of your ML pipeline.  

Most projects could benefit from using TFRecords, but it requires a lot of reading, looking at several coding examples, etc.  Our project aims to help limit the time spent on converting your data over to the TFRecord format so you can take advantage of the speed of TFRecord and start training your models and not waste time elsewhere.  TFDS CLI will give you more control at the cost of more documentation and formatting, but most datasets can be handled in a simpler way, enter smart-tfrecord-writer!

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

If you have any questions, suggestions, or feedback, feel free to reach out:

- Email: [caharper@smu.edu](mailto:caharper@smu.edu)
- GitHub: [caharper](https://github.com/caharper)
