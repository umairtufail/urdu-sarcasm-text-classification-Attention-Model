# Multi-Head Attention LSTM Model for Text Classification

This repository contains Python code for a text classification model using multi-head attention with LSTM layers. The model is implemented using TensorFlow and Keras.

## Description

This project focuses on developing a deep learning model for text classification, specifically designed to identify sarcasm in Urdu tweets. The model utilizes a multi-head attention mechanism along with LSTM layers to capture complex patterns in the text data. By leveraging advanced natural language processing techniques, the model aims to accurately classify Urdu tweets as either sarcastic or non-sarcastic.

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Usage

1. Clone the repository:

```bash
git clone <repository_url>
cd <repository_directory>
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the notebook `text_classification_model.ipynb` in Google Colab or any Jupyter environment.

4. Follow the instructions in the notebook to mount your Google Drive and load the dataset.

5. Train the model by executing the code cells in the notebook.

6. Save the trained model using the provided code snippet in the notebook.

7. Visualize the training results using Matplotlib.

8. Evaluate the model's performance using evaluation metrics such as accuracy, precision, recall, and F1-score.

## File Description

- `text_classification_model.ipynb`: Jupyter notebook containing the code for the text classification model.
- `requirements.txt`: List of Python dependencies required to run the code.

## Dataset

The dataset used for training and evaluation is the **Urdu Sarcastic Tweets Dataset**. It is stored in an Excel file (`urdu_embedding_test_data.xlsx`). The dataset contains text samples and their corresponding labels.

## Model Architecture

The model architecture consists of an Embedding layer followed by two Bidirectional LSTM layers with dropout regularization. Multi-head attention is applied to the LSTM output sequences. Finally, dense layers with activation functions are used for classification.

## Saving the Model

The trained model can be saved in HDF5 format using the provided code snippet in the notebook. The saved model file (`my_improved_model.h5`) will be stored in your Google Drive.

## Evaluation Metrics

After training the model, evaluation metrics such as accuracy, precision, recall, and F1-score are calculated using scikit-learn's `accuracy_score`, `precision_score`, `recall_score`, and `f1_score` functions.

## Visualizing Training Results

Training and validation accuracy as well as training and validation loss are visualized using Matplotlib.
