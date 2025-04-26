# 🧠 Micrograd-Mini

**Micrograd-Mini** is a tiny scalar-valued autograd engine with a basic neural network framework — built from scratch, inspired by [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd).

It supports reverse-mode automatic differentiation (backpropagation) and is capable of training a simple neural network (MLP) on tasks like XOR. Great for learning the internals of backprop!

---

## 🔥 Features

- ✅ Reverse-mode autodiff (backpropagation)
- ✅ Dynamic computation graph (DAG)
- ✅ Custom `Value` class with gradients
- ✅ Tanh and ReLU activations
- ✅ Basic Neuron, Layer, and MLP implementation
- ✅ Trains on XOR dataset

---

## 📁 Project Structure

```
micrograd-mini/
    ├── engine.py # Core autodiff engine (Value class)
    ├── nn.py # Neural network components
    ├── train.py # Training loop for XOR
    ├── example.py # Better example using XOR dataset 
└── README.md # Project info
```

---

## 🚀 Getting Started

### 🔧 Requirements

- Python 3.7+
- No external libraries needed (pure Python)

### ▶️ Run Training

```
python train.py
```
# 📈 Example Output
```
--- Final Predictions after Training ---

Input: [0.0, 0.0] => Predicted: 0.01 | Target: 0.0
Input: [0.0, 1.0] => Predicted: 0.98 | Target: 1.0
Input: [1.0, 0.0] => Predicted: 0.97 | Target: 1.0
Input: [1.0, 1.0] => Predicted: 0.03 | Target: 0.0

Training complete! 🎯
```
# 🧠 Learn by Building
Want to really understand backpropagation and gradients?

+ Dive into engine.py and explore the Value class

+ Inspect how operations dynamically build a graph

+ See how .backward() traverses it for gradient computation — just like real frameworks!
## 🙏 Attribution

This project is heavily inspired by [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy, licensed under the MIT License.

---

## 🪪 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## ✨ Author

Built with ❤️ by **Muawiya**, as part of a deep dive into AI, neural nets, and autodiff fundamentals.

### 🌐 Connect With Me

- 📺 YouTube: [@Coding_Moves](https://www.youtube.com/@Coding_Moves)
- 💻 GitHub: [Muawiya](https://github.com/Muawiya-contact) 
- 🧠 LeetCode: [Moavia_Amir](https://leetcode.com/u/Moavia_Amir/) 
- 📊 Kaggle: [Moavia Amir](https://www.kaggle.com/) 

---
