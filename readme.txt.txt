🍔 Fast Food Image Classification

This project was developed as part of the **BIM447 Introduction to Deep Learning** course.  
It compares two convolutional neural network architectures — a **custom-built SimpleCNN** and a **pretrained DenseNet121** — to classify images of fast food into ten categories.

---

📘 Project Overview

- Goal: Automatically classify 10 types of fast food from images.  
- Dataset: [Fast Food Classification V2 (Kaggle)](https://www.kaggle.com/datasets/utkarshsaxenadn/fast-food-classification-dataset)  
- Members: İsmail Çınar, Şulenur Şahan  
- Framework: PyTorch  

The models were trained and tested on the same dataset to compare performance in terms of accuracy, generalization, and convergence behavior.

---

🧠 Models

🟦 SimpleCNN
A 7-layer custom convolutional network built from scratch.  
It uses batch normalization, ReLU activations, and dropout regularization.  
Designed for interpretability and efficiency on small-scale experiments:contentReference[oaicite:0]{index=0}.

🟩 DenseNet121 (Transfer Learning)
A pretrained **DenseNet121** with the last 45 layers unfrozen for fine-tuning.  
The classifier head was replaced with a new 3-layer fully connected block.  
This model leverages pretrained ImageNet weights for improved generalization:contentReference[oaicite:1]{index=1}.

---

🧩 Data Pipeline

All datasets and transformations are defined in `data_generators.py`:contentReference[oaicite:2]{index=2}.

- **Augmentation:** rotation, flipping, color jitter, random erasing  
- **Normalization:** ImageNet mean/std  
- **Class balancing:** Computed using `sklearn.utils.compute_class_weight`

Data is organized as:

Fast Food Classification V2/
├── Train/
├── Valid/
└── Test/

yaml
Kodu kopyala

---
⚙️ Requirements

```bash
pip install torch torchvision numpy scikit-learn
🚀 How to Run
Clone this repository

bash
git clone https://github.com/Sahansule/fastfood-classification.git
cd fastfood-classification
Prepare dataset
Download and extract the Kaggle dataset into the same folder.

Run dataset preview

bash
python data_generators.py
Train models

python
from simple_cnn import SimpleCNN
from densenet import DenseNetCustom
from data_generators import simple_cnn_train_dataset, valid_dataset
📊 Results Summary
Model	Train Epochs	Test Accuracy	Notes
SimpleCNN	40	Moderate	Slower convergence
DenseNet121	23 (Early Stop)	Higher	Better generalization

DenseNet121 achieved higher and more stable accuracy while SimpleCNN remained useful for lightweight environments
BIM447_Introduction_to_Deep_Learning

.

📚 Reference
PyTorch Documentation – https://pytorch.org/docs

Kaggle Dataset – Fast Food Classification V2

Huang et al., Densely Connected Convolutional Networks (CVPR 2017)

🧑‍💻 Authors:
Şulenur Şahan, İsmail Çınar
Eskişehir Technical University – BIM447 Term Project (2025)