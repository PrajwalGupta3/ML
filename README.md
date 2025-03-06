
# 1)Transfer Learning for Image Classification with CNN and Supervised Models

This project demonstrates a hybrid approach to image classification by combining deep learning with traditional machine learning. It leverages a pre-trained ResNet50 model to extract image features from the CIFAR-10 dataset and then uses various supervised classifiers—namely a Support Vector Machine (SVM), Random Forest, and XGBoost—to perform classification.

The project showcases:
- **Transfer Learning:** Using ResNet50 pretrained on a large dataset to extract rich features.
- **Supervised Learning:** Training classifiers on the extracted features.
- **Model Comparison:** Evaluating the performance of SVM against other classifiers.


## Prerequisites & Setup

Before running the code, ensure that your system is updated and has the required version of Python and packages installed. The first few lines of the code are shell commands that update your package repositories, install Python 3.14 (while checking the version for Python 3.12), and install essential Python libraries quietly:

```bash
!sudo apt update -y
!sudo apt install python3.14
!python3.12 --version
!pip install torch torchvision scikit-learn numpy --quiet
```

These commands ensure your environment is ready for running deep learning experiments.

## Dataset Overview

The project uses the **CIFAR-10 dataset**, which is a widely used collection of 60,000 32x32 color images in 10 classes (with 6,000 images per class). The dataset is split into 50,000 training images and 10,000 testing images.
### Why CIFAR-10?

- **Beginner-Friendly:** It is a well-balanced dataset that is simple enough for educational purposes yet challenging enough to demonstrate the power of deep learning.
- **Standard Benchmark:** CIFAR-10 is a common benchmark in image classification, making it easier to compare results with other models.


## Why ResNet50?

- **Deep Feature Extraction:** ResNet50 is a powerful convolutional neural network that uses residual connections to enable training of very deep networks.
- **Pre-trained on ImageNet:** It has been pre-trained on a massive dataset (ImageNet), which means it has already learned to extract rich and useful features from images.
- **Efficiency:** The architecture is both efficient and effective, making it a great choice for transfer learning.


## Why Use SVM 

- **SVM Advantages:**
  - **High-Dimensional Data Handling:** After feature extraction, the resulting data is high-dimensional. SVMs are known to work well in such spaces.
  - **Robustness:** SVMs can generalize well, especially on smaller to medium-sized datasets.
  
  **Additional Classifiers:
      Random Forest: Provides an ensemble method that aggregates multiple decision trees.
      XGBoost: A gradient boosting method known for its performance and speed.
 
    

## Code Walkthrough

Below is a step-by-step explanation of the code with technical insights and analogies.

### 1. System Setup & Package Installation

The code begins with system-level commands to update packages and install the required Python version and libraries.  
**Analogy:** Setting up your kitchen by ensuring you have the latest utensils and ingredients before starting to cook.


### 2. Importing Libraries and Setting Up the Device

The code imports necessary libraries for deep learning, image transformations, dataset handling, and machine learning. It then sets up the device for computation (GPU if available).

```python
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
from torchvision.models import resnet50
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

**Analogy:** Deciding whether to use an industrial-grade oven (GPU) or a regular one (CPU) based on availability.


### 3. Data Preprocessing and Loading

Images are resized to 224x224 (the input size required by ResNet50) and converted to PyTorch tensors. The CIFAR-10 dataset is then downloaded and loaded into training and testing sets.

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet requires 224x224 images
    transforms.ToTensor()           # Convert images to PyTorch tensors
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
```

**Analogy:** Prepping your ingredients by washing, cutting, and measuring them before cooking.


### 4. Loading the Pre-trained ResNet50

A pre-trained ResNet50 model is loaded and set to evaluation mode. This model is used only for feature extraction, so its final classification layers are not used.

```python
resnet = resnet50(pretrained=True).to(device)
resnet.eval()  # Set model to evaluation mode
print("ResNet50 has been successfully loaded")
```

**Analogy:** Using a master chef’s pre-taught techniques rather than learning from scratch.

### 5. Feature Extraction

The `extract_features` function passes images through the ResNet50 model, extracts the features, and converts them to NumPy arrays for further use by SVM.

```python
def extract_features(loader):
    features, labels = [], []
    with torch.no_grad():  # No need to compute gradients during feature extraction
        for images, targets in loader:
            images = images.to(device)  # Move images to the GPU (if available)
            feats = resnet(images)      # Extract features using ResNet50
            features.append(feats.cpu().numpy())  # Move features to CPU and convert to NumPy
            labels.append(targets.numpy())        # Convert labels to NumPy
    return np.concatenate(features), np.concatenate(labels)  # Combine all features and labels

print("Extracting the features")
start_time = time.time()

train_features, train_labels = extract_features(train_loader)
test_features, test_labels = extract_features(test_loader)

end_time = time.time()
print(f"Feature Extraction Completed in {end_time - start_time:.2f} seconds")
```

**Analogy:** Turning high-resolution photos into simplified sketches that capture only the essential details.


### 6. Training the SVM Classifier

An SVM classifier with a linear kernel is trained on the features extracted from the training dataset.

```python
svm = SVC(kernel='linear')  # Using a linear SVM
svm.fit(train_features, train_labels)
print("SVM Training successful")
```

**Analogy:** Teaching an art critic to identify paintings based on simplified sketches rather than raw, detailed images.



### 7. Evaluating the Model

The SVM classifier is then used to predict labels for the test set, and its accuracy is computed.

```python
preds = svm.predict(test_features)
accuracy = accuracy_score(test_labels, preds)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```

**Analogy:** Grading an exam to see how well the student (the SVM) has learned to classify the images.


### 8. Training Additional Classifiers

For further comparison, the code trains two additional classifiers:
- **Random Forest Classifier:** An ensemble learning method that uses multiple decision trees.
- **XGBoost Classifier:** A gradient boosting model known for high performance.

```python
import joblib

# Saving the trained SVM model
joblib.dump(svm, "my_svm_model.pkl")
print("Model is Saved as 'my_svm_model.pkl'")

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rf.fit(train_features, train_labels)
test_acc = rf.score(test_features, test_labels)
print(f"Random Forest Test Accuracy: {test_acc * 100:.2f}%")

from xgboost import XGBClassifier 
xgb = XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric="logloss")
xgb.fit(train_features, train_labels)
test_acc = xgb.score(test_features, test_labels)
print(f"XGBoost Test Accuracy: {test_acc * 100:.2f}%")
```

**Analogy:** Trying different expert critics to see which one gives the best evaluation based on the same set of sketches.


## How to Run the Code

1. **Clone the Repository:**  
   Clone this repository to your local machine using Git.

2. **Environment Setup:**  
   - Run the provided shell commands to update the system and install the required Python version and libraries.
   - Ensure you have a CUDA-enabled GPU for faster feature extraction (optional).

3. **Run the Notebook:**  
   Open the notebook (or Python script) and run each cell sequentially. The notebook will download the CIFAR-10 dataset, extract features using ResNet50, train classifiers, and display performance metrics.

## Conclusion
  This approach while might not give high accuracy(due to small size of dataset) , leads to lesser training time as compared to a full fledged DNN or CNN. Furthermore, if accuracy can by compromised further , the RandomForest reduces the training time by more than half.XGBoost on the other hand gave a more balanced approach of balncing accuracy and training time(compared to other 2 models). If you are an ML enthusiast too, who learned something useful from this repo, please feel free to experiment further with it , by hyperparameter tuning, trying more such differnt approaches. If you do come up with some ,please feel to reach out to me and tell me about it, as it would lead me to learning new concepts and broaden my horizen.






# 2) Running the model on local device


## Overview

In this project, we:
1. **Load a pre-trained SVM model** (saved as `my_svm_model.pkl`).
2. **Use ResNet50** (pre-trained on ImageNet) as a feature extractor without modifying its final layer, so that its output is a 1000-dimensional feature vector.
3. **Define a function** to process an image: resize, convert it to a tensor, and extract features.
4. **Map the predicted class index** (from CIFAR-10) to a human-readable class name.
5. **Display the image** along with its predicted label.

*Analogy:* Imagine using a master chef’s secret recipe (the SVM model) that was perfected using high-quality ingredients (features from ResNet50). You then take a new ingredient (the input image), process it, and the recipe tells you exactly what dish (class) it is!

## Code Breakdown

### 1. Importing Libraries

```python
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import joblib
from torchvision.models import resnet50
import matplotlib.pyplot as plt
```

- **torch & torchvision:** Provide deep learning tools and utilities.
- **PIL (Python Imaging Library):** Used for image processing.
- **numpy:** Handles numerical operations.
- **joblib:** Loads the pre-trained SVM model.
- **matplotlib:** Displays the image with prediction.

*Analogy:* Think of these as your kitchen appliances and utensils needed to prepare and serve a dish.

### 2. Loading the Trained SVM Model

```python
svm = joblib.load("my_svm_model.pkl")
```

- This loads the SVM model that was trained earlier using ResNet features.
  
*Analogy:* Like retrieving a secret family recipe that has been stored safely, ready to be used to identify a dish.

### 3. Loading ResNet50 for Feature Extraction

```python
model = resnet50(pretrained=True)
model.eval()
```

- **resnet50(pretrained=True):** Loads the ResNet50 model with weights pre-trained on ImageNet.
- **model.eval():** Puts the model in evaluation mode (disables training behaviors like dropout).

*Analogy:* This step is like using a proven cookbook from a master chef whose techniques are trusted and tested.

### 4. Defining the Feature Extraction Function

```python
def extract_features_from_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match ResNet input
        transforms.ToTensor(),          # Convert image to tensor
    ])

    image = Image.open(image_path).convert("RGB")  # Open image
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        features = model(image)  # Extract features using ResNet50

    return features.squeeze().numpy().reshape(1, -1)  # Convert to 2D array
```

- **Transformation:** Resizes the image to 224×224 pixels and converts it into a tensor.
- **Image Loading:** Opens and converts the image to RGB.
- **Batch Dimension:** Adds an extra dimension because PyTorch models expect a batch of images.
- **Feature Extraction:** Passes the image through ResNet50 to get a feature vector.
- **Reshaping:** Converts the output into a 2D array suitable for the SVM.

*Analogy:* Imagine taking a high-resolution photo, resizing it to a standard format, and then using a filter to extract only the most important details.

### 5. Mapping Class Indices to Names

```python
cifar10_classes = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}
```

- This dictionary maps numeric class indices (0–9) to human-readable labels based on the CIFAR-10 dataset.

*Analogy:* Like a legend on a treasure map, where each symbol corresponds to a specific landmark or treasure.

### 6. Processing the Image & Making a Prediction

```python
image_path = "/content/car1.jpeg"
image_features = extract_features_from_image(image_path)
prediction = svm.predict(image_features)
predicted_index = prediction[0]
predicted_class = cifar10_classes.get(predicted_index, "Unknown")

print("Predicted class:", predicted_class)
```

- **Image Path:** Specifies the file path of the image to classify.
- **Feature Extraction:** Extracts features from the image using the defined function.
- **Prediction:** The SVM model predicts the class index.
- **Mapping:** Converts the predicted index into a class name using the dictionary.
- **Output:** Prints the predicted class.

*Analogy:* Like feeding ingredients into a machine that tells you what dish you have based on a secret recipe.

### 7. Displaying the Image with Prediction

```python
image = Image.open(image_path).convert("RGB")
plt.imshow(image)
plt.title("Predicted: " + predicted_class)
plt.axis('off')
plt.show()
```

- **Image Loading:** Reopens the image for display.
- **Display:** Uses Matplotlib to display the image with the predicted class as the title.
- **Formatting:** Turns off axis labels for a cleaner view.

*Analogy:* It’s like presenting your finished dish on a plate with a label that proudly shows its name.

## How to Run the Code

1. **Clone the Repository:**  
   Clone this repository to your local machine.

2. **Install Dependencies:**  
   Ensure you have the required libraries installed:
   ```bash
   pip install torch torchvision pillow numpy joblib matplotlib
   ```

3. **Place the Necessary Files:**  
   - Make sure `my_svm_model.pkl` (the trained SVM model) is in the repository directory.
   - Update `image_path` in the code with the path to the image you want to classify.

4. **Run the Script:**  
   Execute the script using Python:
   ```bash
   python your_script.py
   ```
   The code will print the predicted class and display the image with its predicted label.

## Conclusion

This project illustrates a practical application of transfer learning by using a pre-trained ResNet50 model to extract features and an SVM model to classify a single image. By combining deep learning and traditional machine learning techniques, you can build efficient and modular image classification systems. This approach not only helps in understanding both methodologies but also provides flexibility in experimenting with different models.
