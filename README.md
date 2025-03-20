# Cat vs. Dog Image Classification with SVM

This project implements an image classification model to distinguish between images of cats and dogs using a Support Vector Machine (SVM) classifier from scikit-learn.

## Dataset

* The dataset used is the "Dogs vs. Cats Images" dataset from Kaggle, available at: [https://www.kaggle.com/datasets/chetankv/dogs-cats-images?resource=download&select=dataset](https://www.kaggle.com/datasets/chetankv/dogs-cats-images?resource=download&select=dataset)
* The dataset is organized into training and testing sets within the `dataset` folder.
* The project uses the training set folder to train and test the model.
* The dataset is downloaded using the `opendatasets` library.

## Dependencies

* `opendatasets`
* `pandas`
* `numpy`
* `seaborn`
* `matplotlib`
* `opencv-python` (`cv2`)
* `scikit-image` (`skimage`)
* `scikit-learn` (`sklearn`)

To install these dependencies, run:

```bash
pip install opendatasets pandas numpy seaborn matplotlib opencv-python scikit-image scikit-learn
```
Project Structure

    README.md: This file.
    The project consists of a single python script that performs the following steps:
        Downloads the dataset using opendatasets.
        Loads and preprocesses the images from the training set.
        Resizes images to 50x50 pixels and converts them to grayscale.
        Flattens the images into feature vectors.
        Splits the data into training and testing sets.
        Trains an SVM model using the training data.
        Evaluates the model's performance on both the training and testing sets.
        Prints the accuracy scores.

Code Description

The python code performs the following actions:

    Dataset Download: Downloads the dataset using opendatasets.
    Data Loading and Preprocessing:
        Iterates through the dogs and cats subdirectories in the training set.
        Loads images in grayscale using cv2.imread.
        Resizes images to 50x50 pixels using cv2.resize.
        Flattens the images into feature vectors using numpy.array().flatten().
        Stores the feature vectors and labels in lists.
    Data Splitting:
        Splits the data into training and testing sets using train_test_split with a test size of 20% and a random state of 77.
    Model Training:
        Trains an SVM model with a polynomial kernel, C=1, and gamma='auto'.
        Fits the model to the training data.
    Model Evaluation:
        Predicts labels for the training and testing sets.
        Calculates and prints the accuracy scores for both sets using accuracy_score and model.score.
        Prints the predicted values for the training and testing sets.

Running the Code

    Install the required dependencies.
    Run the python script.

Bash

python your_script_name.py

    The script will download the dataset, train the model, and print the accuracy scores.

Results

The script outputs the following:

    Length of the data list.
    Example of a data point (flattened image and label).
    Length of the labels list.
    Example of a feature vector.
    Predicted values for the training and testing sets.
    Accuracy of the training data.
    Accuracy of the testing data.

The model's performance can be assessed by the reported accuracy scores.
Improvements

    Hyperparameter tuning using GridSearchCV for better model performance.
    Using more advanced feature extraction techniques.
    Adding data augmentation to increase dataset size and diversity.
    Implementing cross validation.
    Adding classification reports and confusion matrices for more detailed evaluation.
    Saving the trained model to a file for later use.
