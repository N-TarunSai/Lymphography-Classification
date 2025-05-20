🧬 Lymph Node Tumor Classification using Deep Learning
This project presents a deep learning solution for classifying histopathological images of lymph nodes as benign or malignant. By leveraging transfer learning on a pre-trained convolutional neural network, this work aims to support early detection and diagnosis of cancer, particularly metastatic breast cancer, in lymph node tissue.

🚀 Project Overview
⏱️ Objective: Predict cancer presence in lymph node histopathology slides using image classification.

🧠 Approach: Fine-tune EfficientNetB0 using transfer learning and apply augmentation to improve model robustness.

📈 Outcome: Achieved high training and validation accuracy with a focus on generalization through regularization techniques.

📂 Dataset Details
📚 Source: TensorFlow Datasets — PatchCamelyon (PCam)

🔍 Description: The dataset is derived from the Camelyon16 challenge and consists of small (96×96) patches extracted from histopathology scans.

📊 Classes: Binary classification

0: No Tumor

1: Tumor Present

🔢 Size: Over 300,000 labeled image patches (we selected a manageable subset for training/validation/testing)

🧠 Model Architecture
🔧 Base Model: EfficientNetB0 (with pre-trained ImageNet weights)

🔀 Layers Added:

Global Average Pooling

Dense Layer with ReLU Activation

Dropout Layer for Regularization

Final Dense Layer with Sigmoid Activation for Binary Classification

❄️ Freezing: The base model was frozen initially and later unfrozen (fine-tuned) for further improvement.

🛠️ Key Techniques Used
Data Augmentation (horizontal/vertical flip, rotation, zoom)

EarlyStopping and ReduceLROnPlateau callbacks

Adam optimizer with binary cross-entropy loss

Accuracy as the evaluation metric

📊 Model Performance
✅ Training Accuracy: ~99%

✅ Validation Accuracy: ~97%

📉 No signs of overfitting due to effective augmentation and regularization

🧪 Tested on unseen test samples with accurate predictions

🖼️ Sample Predictions
Predictions are visualized with the image patch and its predicted label (tumor / no tumor), allowing manual verification.

📁 Project Structure
├── lymph.ipynb
├── README.md
├── dataset/
│ ├── train/
│ ├── test/
│ └── val/
└── models/ (optional: saved models/checkpoints)

📌 Requirements
Python 3.8+

TensorFlow 2.x

NumPy

Matplotlib

Scikit-learn

TensorFlow Datasets

You can install the required packages using:

pip install -r requirements.txt

🚧 Future Work
Deploy model via Flask web app for real-time classification

Integrate Grad-CAM for visual explanations

Expand to multi-class tumor subtype classification

👨‍💻 Author
Tarun Sai Nyalakanti
B.Tech in Computer Science, VIT-AP University
AWS Cloud Certified | MERN Stack Developer | AI/ML Enthusiast

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
