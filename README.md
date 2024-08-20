# CIFAR-10 Image Classification using Fastai and ResNet152

This project focuses on classifying images from the CIFAR-10 dataset using deep learning. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes. The project employs the Fastai library and a pre-trained ResNet152 model to achieve high classification accuracy.

## Project Overview

- **Dataset**: CIFAR-10, consisting of 50,000 training images and 10,000 test images, split across 10 classes.
- **Data Preparation**: The dataset is extracted, split into training and validation sets, and organized into folders based on their classes.
- **Model Architecture**: ResNet152, a deep convolutional neural network pre-trained on ImageNet, is fine-tuned for the CIFAR-10 classification task.
- **Training**: The model is trained using a combination of data augmentation, transfer learning, and learning rate scheduling.
- **Evaluation**: The model's performance is evaluated on the validation set and predictions are made on the test set.
- **Submission**: The final predictions are formatted and saved in a CSV file for submission.

## Requirements

- Python 3.7+
- Fastai
- PyTorch
- NumPy
- Pandas
- py7zr

## Installation

1. **Clone the repository:**

2. **Install dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

3. **Download the CIFAR-10 dataset:**
   - Place the CIFAR-10 dataset in the `input` directory with the following structure:
   
    ```
    input/
    └── cifar-10/
        ├── train.7z
        ├── test.7z
        ├── trainLabels.csv
        └── sampleSubmission.csv
    ```

## Running the Project

### Data Preparation

1. **Extract the Dataset:**
   - The CIFAR-10 dataset is provided in `.7z` format, which is extracted using the `py7zr` library.

2. **Organize Images:**
   - Images are organized into training and validation sets by creating folders for each class. Approximately 80% of the images are used for training, and 20% for validation.

### Model Training

1. **Data Loading:**
   - The `ImageDataLoaders` class from Fastai is used to load images from the training and validation folders. Data augmentation is applied during loading to improve model generalization.

2. **Model Definition:**
   - The ResNet152 model, pre-trained on ImageNet, is used as the backbone of the model. The final layers are fine-tuned to adapt to the CIFAR-10 classification task.

3. **Training:**
   - The model is trained in two phases:
     1. **Initial Training**: The model is trained with frozen early layers and a learning rate finder is used to identify the optimal learning rate.
     2. **Fine-Tuning**: The model is then unfreezed, and additional training is performed with lower learning rates to fine-tune the entire network.

4. **Save the Model:**
   - After training, the model is saved for later use in inference.

### Model Inference

1. **Extract Test Dataset:**
   - The test dataset is extracted and prepared for inference.

2. **Load the Model and Predict:**
   - The saved model is loaded, and predictions are made on the test images. The predictions are converted to class labels based on the CIFAR-10 classes.

3. **Generate Submission:**
   - The predictions are formatted into a CSV file (`submission.csv`) for submission. The file contains the image IDs and the corresponding predicted labels.

## Example Output

- **Training Output:**

    ```
    epoch	train_loss	valid_loss	accuracy	time
    0	0.361632	0.227418	0.925226	06:16
    1	0.202388	0.149157	0.948394	06:16
    ...
    ```

- **Submission File:**

    ```csv
    id,label
    287200,automobile
    33557,cat
    281872,deer
    ...
    253814,truck
    17297,dog
    259315,deer
    ```
