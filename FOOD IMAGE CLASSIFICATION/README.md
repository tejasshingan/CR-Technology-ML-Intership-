# Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Dependencies](#dependencies)
4. [Training the Model](#training-the-model)
5. [Making Predictions](#making-predictions)
6. [Visualization](#visualization)
7. [License](#license)

## Introduction<a name="introduction"></a>

This project is part of the Coding Raja Technologies Machine Learning Internship and focuses on image classification for food recognition using the Inception-v3 model. The model is trained on the Food-101 dataset, which contains images of various food items.

## Dataset<a name="dataset"></a>

You can download and extract the Food-101 dataset using the following commands:

```bash
!wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
!tar xzvf food-101.tar.gz
```

## Dependencies<a name="dependencies"></a>

Make sure to install the required dependencies:

```bash
pip install tensorflow keras numpy matplotlib opencv-python-headless pillow
```

## Training the Model<a name="training-the-model"></a>

To train the Inception-v3 model on the Food-101 dataset, you can run the provided Jupyter Notebook `Food_Classification.ipynb`. This notebook covers data preprocessing, model training, and evaluation.

## Making Predictions<a name="making-predictions"></a>

You can use the trained model to make predictions on food images. To do this, load the model using:

```python
from tensorflow.keras.models import load_model
model = load_model('best_model_101class.hdf5', compile=False)
```

Then, use the `predict_class` function to classify images.

## Visualization<a name="visualization"></a>

The project includes visualization of model performance with training and validation accuracy and loss plots. Additionally, you can visualize activation layers, heatmaps, and class activation maps using provided functions in the notebook.

Feel free to explore and modify the code as needed for your own food recognition projects.

## License<a name="license"></a>

This project is licensed under the [MIT License](LICENSE).
```
