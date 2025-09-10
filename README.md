# GUVI_PROJECT2: Handwritten Digit Recognizer

📌 Problem Statement
Manual recognition of handwritten digits is time-consuming and error-prone, especially when processing large volumes of data like scanned forms or bank checks. A deep learning model can automate and improve the accuracy of handwritten digit classification.

❓ Question
How can we build a deep learning model using TensorFlow that accurately recognizes handwritten digits (0–9) from images?

📖 Summary
This project uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify digits (0–9) from the MNIST dataset. The dataset contains 70,000 grayscale images of handwritten digits (60,000 for training, 10,000 for testing), each of size 28×28 pixels.
The model will be trained to learn patterns in handwritten digits and then predict digits from unseen data.

## 🛠️ Methodology
### 1. Data Preparation
- Load the MNIST dataset using tensorflow.keras.datasets.
- Normalize pixel values (0–255 → 0–1) for better training stability.
- Reshape inputs to (28, 28, 1) for CNN compatibility.
- Introduce noise into the dataset.
- Compare accuracies between Vanilla loss function and APL loss function over different noise levels.

### 2. Model Architecture (CNN)
- Conv2D layers – extract features from the images.
- MaxPooling2D layers – reduce spatial dimensions.
- Flatten layer – convert 2D features into 1D vector.
- Dense layers – fully connected layers for classification.
- Output layer – softmax activation with 10 units (digits 0–9).

### 3. Training
- Compile model with:
  - Optimizer: Adam
  - Loss:
      - Sparse Categorical Crossentropy
      - API Framework combining Sparse Categorical Crossentropy(Active Loss) and Reverse Crossentropy(Passive Loss)
  - Metric: Accuracy
- Train for 10 epochs with validation on test set.

### 4. Evaluation
- Confusion matrix to analyze misclassifications.
- Accuracy score on test data.
- Visualization of predictions with matplotlib.

## 📊 Results
- Test Accuracy:
    - On Sparse Categorical Crossentropy: ~99%
    - On APL Framework: ~98%
 
Confusion matrix shows most predictions are correct, with minor misclassifications between visually similar digits (e.g., 4 vs 9, 5 vs 6).

## ⚙️ Technologies Used
- Python 3.10
- TensorFlow / Keras – for building and training CNN
- NumPy – for numerical operations
- Matplotlib – for visualization (predictions, accuracy plots)
- scikit-learn – for confusion matrix and evaluation metrics

## 🚀How to run
1. Download or Clone this Repository.
2. Open the .ipynb notebook on Google Colab.
3. Install Dependencies(pip install tensorflow numpy matplotlib scikit-learn).
4. Run the notebook.
