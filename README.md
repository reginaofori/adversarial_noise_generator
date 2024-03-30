## Adversarial Noise Generator

The Adversarial Noise Generator is a Python library designed to introduce adversarial noise into images. This noise is crafted to trick pre-trained image classification models into misclassifying the altered image as a specified target class, while maintaining the original image's appearance to human viewers.


### Project Structure

Below is the structure of the  project:

```bash
adversarial-noise-generator/
│
├── adversarial/            # Core library code
│   ├── __init__.py         # Makes adversarial a Python package
│   ├── models.py           # Model loading and processing
│   ├── attacks.py          # Adversarial attack implementations (FGSM)
│   └── utils.py            # Utility functions for image processing and others
│
├── examples/               # Example scripts and notebooks
│   ├── generate_example.py # Script to generate an adversarial image
│   └── visualize_attacks.ipynb # Jupyter notebook for simple illustration
│
├── tests/                  # Unit and integration tests
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_attacks.py
│   └── test_utils.py
│
├── setup.py                # Setup script for installing the library
├── requirements.txt        # List of required project dependencies
├── dev-requiremnt.txt      # List of extra dependencies, necessary during test
├── README.md               # Project overview, installation instructions, and usage examples
└── .gitignore              # untracked files to ignore
```


### Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Generating Adversarial Images](#generating-adversarial-images)
  - [Jupyter Notebook](#visualizing-attacks)
  - [Limitation](#limitation)
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

To generate adversarial images using the library, run the `generate_examples.py` script which takes in the ``image`` and ``the target class id``. For a list of target class IDs and class names as used in ImageNet, please refer to the official [ImageNet class list](https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/).


```bash
python examples/generate_examples.py [path/to/image.jpg-needed] [208 (target-class-index-needed)] 
e.g. python ./examples/generate_examples.py ./examples/sample_images/bee.jpg 208 --epsilon 0.02 --model_name resnet18
```
- ### Jupyer Notebook

A simple jupyter notebook for illustration

```bash
jupyter notebook examples/attacks_example.ipynb

```

### Limitations

While this Adversarial Noise Generator aims to be a tool for generating adversarial images and testing machine learning model robustness, here is a sumamry of some of its limitations:

- **Model Compatibility:** The current version of the library is primarily tested with specific models pre-trained models and not SOTA models such as transformers/Large Language models (LLMs).

- **Attack Techniques:** The library includes implementations for popular adversarial attack techniques like FGSM and PGD.

- **Target class:** The target class in the form of json file used consist of only the first 500 target classes in IMAGENET.

### Tests

1.  To run the tests, install the extra dependecies in the dev-requirement.txt 

```bash
pip install -e .[dev]
```

2.  The test folder is run with the following:
```bash
python -m pytest test/
```