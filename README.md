# Password Generation Using AI
# AI-Based Password Generation & Strength Classification System

An advanced AI-driven system that leverages deep learning models to generate secure passwords and classify password strength. This project combines **RNNs, CNNs, and Variational Autoencoders (VAE)** to analyze password patterns, assess complexity, and produce strong, policy-compliant passwords for cybersecurity applications.

---

##  Project Overview

This project aims to improve password security by applying machine learning techniques to:
- Learn real world password patterns  
- Classify password strength (weak, medium, strong)  
- Generate high-entropy passwords using a VAE-based model  

The system is designed for use in **password management tools, authentication systems, and cybersecurity platforms** to reduce the risk of weak or reused passwords.

---

## Key Features

- Password Strength Classification using CNN & RNN models  
- Secure Password Generation using a Variational Autoencoder (VAE)  
- Feature-engineered dataset for improved training accuracy  
- Data preprocessing and normalization pipeline  
- Model evaluation with accuracy and loss tracking  
- Modular architecture for easy model upgrades  
- Scalable design for integration into real-world security systems  

---

## Tech Stack

- **Programming:** Python  
- **Deep Learning:** PyTorch  
- **Models:** RNN, CNN, Variational Autoencoder (VAE)  
- **Data Processing:** Pandas, NumPy  
- **Visualization:** Matplotlib  
- **Environment:** Jupyter Notebook / Python Scripts  

---

## 📂 Project Structure

```text
Password-Machine-Learning/
├── data/                 # Sample datasets (small only)
├── preprocessing/        # Feature engineering scripts
├── models/
│   ├── rnn_model.py
│   ├── cnn_model.py
│   └── vae_model.py
├── training/
│   ├── train_rnn.py
│   ├── train_cnn.py
│   └── train_vae.py
├── evaluation/
│   └── model_metrics.ipynb
├── notebooks/
│   └── experiments.ipynb
├── requirements.txt
└── README.md

