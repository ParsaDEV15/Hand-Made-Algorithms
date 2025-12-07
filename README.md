# 🛠 Hand-Made-Algorithms

This repository contains **manually implemented machine learning algorithms**, built from scratch without using ready-made ML library models.  
This project focuses on deeply understanding the mathematical logic and training mechanics behind core ML algorithms.

---

## 📁 Current Algorithms Implemented

Inside the `models/` directory, two fundamental models are fully implemented:

| Algorithm | Optimizer Used | Purpose |
|---|---|---|
| **Linear Regression (SGD)** | Stochastic Gradient Descent | Predicts continuous numeric values by minimizing Mean Squared Error |
| **Logistic Regression (BSGD)** | Batch Stochastic Gradient Descent | Binary classification using Sigmoid + Cross-Entropy loss |

---

## 📂 Project Structure

```
Hand-Made-Algorithms/
│
├── models/
│   ├── linear_regression_sgd.py       # Linear Regression w/ SGD update rule
│   ├── logistic_regression_bsgd.py    # Logistic Regression w/ Batch SGD
│
├── tests/                             # (future) benchmark & accuracy checks
├── README.md
└── examples/                          # planned for sample demo usage
```

---

## 🧠 Why This Project Exists?

✔ Learn how ML algorithms actually optimize weight values  
✔ Remove abstraction — implement formulas **manually**  
✔ Strengthen mathematical intuition  
✔ Help future scaling to more models (SVM, KNN, PCA, NN from scratch...)  

This repo is a hands-on learning playground for building models at the foundational gradient level.

---

## 🔮 Future Planned Implementations

| Upcoming Algorithm | Status |
|---|---|
| SVM | ⏳ Planned |
| KNN | ⏳ Planned |
| Decision Tree | ⏳ Planned |
| PCA | ⏳ Planned |
| Neural Network from Scratch | 🔥 Definitely coming |

---

### ⭐ If this repository interests you, consider giving it a Star!
