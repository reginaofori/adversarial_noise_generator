## Adversarial Noise Generator

The Adversarial Noise Generator is a Python library designed to introduce adversarial noise into images. This noise is crafted to trick pre-trained image classification models into misclassifying the altered image as a specified target class, while maintaining the original image's appearance to human viewers.


### Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Generating Adversarial Images](#generating-adversarial-images)
  - [Visualizing Attacks](#visualizing-attacks)
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


4. To run the tests, install the extra dependecies in the dev-requirement.txt 

```bash
pip install -e .[dev]
```