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
After 1000 training epochs, you’ll see predictions for XOR:
```
[0.0, 0.0] => 0.01
[0.0, 1.0] => 0.98
[1.0, 0.0] => 0.97
[1.0, 1.0] => 0.03
```
# 🧠 Learn by Building
Want to really understand backpropagation and gradients? Dive into engine.py and see how the Value class works. You’ll see the entire gradient graph built and traversed by hand — just like how real autodiff frameworks work under the hood!

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
