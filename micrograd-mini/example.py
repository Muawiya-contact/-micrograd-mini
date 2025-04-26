# example.py
"""
Example usage of Micrograd-Mini with light training
Author: Muawiya
Inspired by micrograd by Andrej Karpathy
"""

from engine import Value
from nn import MLP

# 1. Create a small MLP
model = MLP(nin=2, nouts=[4, 1])

# 2. Define the XOR dataset
examples = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
]

# 3. Training loop
print("\n--- Training on XOR dataset ---\n")
for epoch in range(1000):
    # Forward pass: compute loss
    loss = 0.0
    for inputs, label in examples:
        input_values = [Value(x) for x in inputs]
        prediction = model(input_values)[0]
        loss += (prediction - label) ** 2  # Mean Squared Error

    # Zero the gradients
    for p in model.parameters():
        p.grad = 0.0

    # Backward pass: compute gradients
    loss.backward()

    # Gradient descent step
    for p in model.parameters():
        p.data += -0.05 * p.grad  # learning rate = 0.05

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.data:.4f}")

# 4. Final predictions
print("\n--- Final Predictions after Training ---\n")
for inputs, label in examples:
    input_values = [Value(x) for x in inputs]
    output = model(input_values)[0]
    print(f"Input: {inputs} => Predicted: {output.data:.4f} | Target: {label}")

print("\nTraining complete! 🎯")
