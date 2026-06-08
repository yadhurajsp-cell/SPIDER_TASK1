# 👕 FashionMNIST Classification with Multi-Branch Neural Network

A deep learning project built using PyTorch that classifies clothing images from the FashionMNIST dataset using a custom multi-branch feedforward neural network architecture with skip connections.

---

## 🚀 Overview

This project demonstrates how to build, train, validate, and evaluate a custom neural network for image classification.

Instead of using a standard fully connected architecture, the model introduces:

* Multi-branch feature extraction
* Skip connections
* Validation-based checkpointing
* GPU acceleration (CUDA support)
* Performance visualization using Matplotlib

The model is trained on the FashionMNIST dataset, which contains grayscale images of clothing items across 10 categories.

---

## 📊 Dataset

FashionMNIST consists of:

* 60,000 training images
* 10,000 testing images
* Image size: 28 × 28 pixels
* 10 clothing categories

Classes include:

| Label | Category      |
| ----- | ------------- |
| 0     | T-Shirt / Top |
| 1     | Trouser       |
| 2     | Pullover      |
| 3     | Dress         |
| 4     | Coat          |
| 5     | Sandal        |
| 6     | Shirt         |
| 7     | Sneaker       |
| 8     | Bag           |
| 9     | Ankle Boot    |

---

## 🏗️ Model Architecture

Input Image (28×28)

↓

Flatten Layer

↓

Hidden Layer (784 → 16)

↓

├── Left Branch (16 → 8 → 8)

│ └── Skip Connection

│

└── Right Branch (16 → 12 → 8)

↓

Concatenation

↓

Output Layer (16 → 10)

↓

Softmax Classification

### Key Features

* ReLU Activations
* Skip Connections
* Branch-based Feature Learning
* Adam Optimizer
* Cross Entropy Loss

---

## 🛠️ Tech Stack

* Python
* PyTorch
* Torchvision
* NumPy
* Pandas
* Matplotlib
* CUDA (optional)

---

## 📂 Project Structure

```text
FashionMNIST-Classifier/
│
├── data_Fashion/
│
├── best_model.pkl
│
├── submission.csv
│
├── train.py
│
├── requirements.txt
│
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/fashionmnist-classifier.git

cd fashionmnist-classifier
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

```bash
python train.py
```

The script will:

1. Download FashionMNIST
2. Train the neural network
3. Validate after each epoch
4. Save the best model
5. Plot training & validation losses
6. Evaluate on the test dataset
7. Generate a submission.csv file

---

## 📈 Training Features

### Validation Monitoring

The best-performing model is automatically saved using validation loss.

```python
if epoch_val_loss < best_model:
```

### Model Checkpointing

```python
pickle.dump(model.state_dict(), f)
```

### Loss Visualization

Training and validation loss curves are plotted after training.

---

## 📊 Evaluation

The model computes:

* Test Accuracy
* Predicted Labels
* CSV Submission File

Example:

```text
Accuracy = 88.42%
```

---

## 📌 Future Improvements

* Convolutional Neural Networks (CNNs)
* Residual Blocks
* Batch Normalization
* Dropout Regularization
* Hyperparameter Tuning
* TensorBoard Integration
* Streamlit Deployment

---

## 🧠 What I Learned

Through this project I gained hands-on experience with:

* Custom Neural Network Design
* Skip Connections
* Data Loading Pipelines
* Model Training Loops
* Validation Strategies
* Checkpoint Saving
* GPU Training with CUDA
* Performance Visualization

---

## 👨‍💻 Author

Yadhuraj S.P.

If you found this project useful, feel free to ⭐ the repository and connect with me.
