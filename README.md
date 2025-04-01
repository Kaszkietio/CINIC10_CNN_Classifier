# CNN architectures exploration on CINIC-10

Authors:

[Błażej Misiura](https://github.com/blazej-misiura)\
[Wojciech Kutak](https://github.com/Kaszkietio)

This project was conducted as part of the "Deep Learning" course at the Warsaw University of Technology. The objective was to develop a convolutional neural network capable of classifying images from the CINIC-10 dataset. This dataset comprises 270,000 images, each sized at 32x32 pixels and categorized into 10 classes. Derived from the CIFAR-10 dataset, CINIC-10 retains the same classes but with resized images. The dataset is partitioned into three subsets: training, validation, and test, containing 90,000 images each. The CINIC10 dataset is an extension of CIFAR-10, incorporating images from both CIFAR and ImageNet. It offers a more extensive and diverse set of images, totaling 270,000 samples evenly distributed across 10 classes.


## Repository Structure

- `notebooks/`: Contains Jupyter notebooks used for model development and experimentation.
- `src/`: Includes source code for model architecture, training routines, and utility functions.
- `data/`: Directory designated for dataset storage and organization.
- `checkpoints/`: Stores model checkpoints saved during training.
- `plots/`: Contains visualizations and plots generated during analysis.

## Installation

To set up the environment and install necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage
Training the Model
To train the CNN model, execute the training script with the desired configuration:

```bash
python src/train.py --config training_config.json
```

The training_config.json file contains hyperparameter settings and other configurations for training.

## Results
The model's performance is evaluated using standard metrics such as accuracy and loss. Detailed results, including plots and analysis, can be found in the notebooks/ and plots/ directories.

## License
This project is licensed under the Apache-2.0 License. See the LICENSE file for more details.
