## Adversarial Noise Generator

The Adversarial Noise Generator is a Python library designed to introduce adversarial noise into images. This noise is crafted to trick pre-trained image classification models into misclassifying the altered image as a specified target class, while maintaining the original image's appearance to human viewers.


### Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Generating Adversarial Images](#generating-adversarial-images)
  - [Jupyter Notebook](#visualizing-attacks)
  - [Limitation] (#Limitation)
- [Tests](#test)


### Features

- Supports loading pre-trained models from torchvision.
- Implements Fast Gradient Sign Method (FGSM) for generating adversarial noise.
- Includes utilities for image preprocessing and postprocessing.
- Provides examples and test suite.

### Installation

Below are the following to install the Adversarial Noise Generator:

1. Clone the repository:
```bash
git clone https://github.com/yourusername/adversarial-noise-generator.git
```

2. Navigate to the cloned directory

```bash
cd adversarial-noise-generator
```

3.  Install the required dependencies

```bash
pip install e .
``` 

### Usage

- ### Generating Adversarial Images

To generate adversarial images using the library, run the `generate_examples.py` script:

```bash
python examples/generate_examples.py [path/to/image.jpg-needed] [208 (target-class-needed)] 
e.g. python ./examples/generate_examples.py ./examples/sample_images/bee.jpg 208 --epsilon 0.02 --model_name resnet18
```
- ### Jupyer Notebook



### Tests

1.  To run the tests, install the extra dependecies in the dev-requirement.txt 

```bash
pip install -e .[dev]
```

2.  The test folder is run with the following:
```bash
python -m pytest test/
```