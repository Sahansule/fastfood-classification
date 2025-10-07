# fastfood-classification
🍔 Fast Food Classification – Deep Learning Project

A deep learning project that classifies images of **10 different fast food categories** using **Convolutional Neural Networks (CNNs)** and **Transfer Learning**.  
The project compares a custom-built **SimpleCNN** model and a **DenseNet121** model fine-tuned on the same dataset to explore performance differences between training from scratch and transfer learning.

# 🍔 Fast Food Classification – Deep Learning Project  

A deep learning project that classifies images of **10 different fast-food categories** using **Convolutional Neural Networks (CNNs)** and **Transfer Learning**.  
The project compares a custom-built **SimpleCNN** model and a **DenseNet121** model fine-tuned on the same dataset to explore performance differences between training from scratch and transfer learning.  

---

## 🧠 Project Proposal
The goal of this project is to develop an image classification system capable of recognizing fast-food types such as **burger, fries, taco, pizza, donut**, etc.  
The dataset used is [Fast Food Classification V2 (Kaggle)](https://www.kaggle.com/datasets/utkarshsaxenadn/fast-food-classification-version-2), containing 20,000 labeled images divided into training, validation, and test sets.

---

## ⚙️ Technologies Used
- **Framework:** PyTorch & Torchvision  
- **Language:** Python (3.10+)  
- **Libraries:** NumPy, scikit-learn, Matplotlib, Pillow, tqdm  
- **Training Setup:** Data augmentation, Label smoothing (0.1), Early stopping, OneCycleLR scheduler  
- **Environment:** Google Colab / VS Code / Jupyter Notebook  

---

## 📊 Key Results
| Model | Epochs | Validation Accuracy | Notes |
|:------|:-------:|:-------------------:|:------|
| **SimpleCNN** | 40 | ~80 % | Gradual learning, slightly less stable |
| **DenseNet121** | 23 (Early Stop) | ~88 % | Faster convergence, better generalization |

DenseNet121 achieved higher and more stable accuracy while SimpleCNN remained useful for lightweight experiments.  
Transfer learning significantly improved performance, especially on visually similar classes (like *Taco vs. Taquito*).

---

## 👩‍💻 Authors
**Şulenur Şahan**, **İsmail Çınar**  
*Eskişehir Technical University – BIM447 Introduction to Deep Learning (2025)*
