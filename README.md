# MNIST Digit Classification using CNN

This project demonstrates the use of a Convolutional Neural Network (CNN) for image classification on the MNIST dataset. The model is built using Keras and TensorFlow, and it classifies handwritten digits (0–9). The project includes data preprocessing, model training, validation, and evaluation using various metrics such as accuracy and confusion matrix.

## Dataset

The MNIST dataset consists of 70,000 grayscale images of handwritten digits (28x28 pixels). These images are split into:
- **60,000 training samples**
- **10,000 test samples**

The dataset is divided into training and validation sets in a 6:1 ratio (55,000 training, 5,000 validation). Each image corresponds to a digit between 0 and 9.

## Project Workflow

### 1. **Data Loading and Preprocessing**
   - Load the MNIST dataset using `keras.datasets.mnist`.
   - Reshape the data to fit the input shape required by CNN (28x28x1).
   - Convert class vectors into one-hot encoded matrices.
   - Normalize the pixel values (0 to 255) to the range [0, 1] for better performance.

### 2. **Model Architecture**
   The CNN model is constructed as follows:
   - **Convolutional Layers**:
     - First layer: `Conv2D` with 32 filters and 3x3 kernel size, followed by ReLU activation.
     - Second layer: `Conv2D` with 64 filters and 3x3 kernel size, followed by ReLU activation.
   - **Pooling Layer**: MaxPooling layer with a 2x2 pool size.
   - **Dropout Layer**: Dropout with 25% probability to prevent overfitting.
   - **Fully Connected Layers**:
     - Flatten the output from the convolutional layers.
     - Dense layer with 256 units and ReLU activation.
     - Another Dropout layer with a 50% probability.
   - **Output Layer**: A Dense layer with 10 units and softmax activation to classify into 10 categories (digits 0–9).

### 3. **Model Compilation**
   The model is compiled using:
   - **Loss function**: Categorical Crossentropy (for multi-class classification).
   - **Optimizer**: Adadelta.
   - **Metrics**: Accuracy.

### 4. **Training the Model**
   The model is trained using the following configurations:
   - **Batch Size**: 128
   - **Epochs**: 50
   - **Callbacks**:
     - **ModelCheckpoint**: Saves the best model based on validation accuracy.
     - **EarlyStopping**: Stops training if the validation loss doesn't improve for 5 epochs.
   - The training data is shuffled to improve generalization.

### 5. **Model Evaluation**
   - After training, the model is evaluated on both validation and test sets.
   - Accuracy and loss metrics are computed.
   - A confusion matrix and classification report are generated to visualize model performance.

### 6. **Saving the Model**
   The trained model is saved as `mnist.h5` for future use.

### 7. **Prediction**
   A `predict_digit` function is defined to predict the digit of a given image.

### 8. **Evaluation on New Images**
   The model can predict new digits by preprocessing the image (resizing and reshaping to 28x28x1) and passing it to the `predict_digit` function.

## Results

- **Training Accuracy**: ~91.58%
- **Validation Accuracy**: ~94.04%
- **Test Accuracy**: ~92.40%
  
   The model demonstrates good performance on both the validation and test datasets, with precision and recall values above 90% for most digits.

### Confusion Matrix
A confusion matrix is generated to analyze the model’s performance across different classes (digits).

### Classification Report
A detailed classification report is generated showing precision, recall, F1-score, and support for each class.

## Conclusion

This project demonstrates how a simple CNN can be used to classify images of handwritten digits. The model performs well on both validation and test data, making it suitable for digit recognition tasks.
