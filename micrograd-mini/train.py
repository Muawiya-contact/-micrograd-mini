# train.py
from nn import MLP
from engine import Value

mlp = MLP(2, [4, 4, 1])

# XOR dataset
data = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
]

for epoch in range(1000):
    total_loss = 0.0
    for x, y in data:
        inputs = [Value(xi) for xi in x]
        pred = mlp(inputs)[0]
        target = Value(y)
        loss = (pred - target) ** 2

        # Reset gradients
        for p in mlp.parameters():
            p.grad = 0.0
        loss.backward()

        # Update parameters
        for p in mlp.parameters():
            p.data -= 0.1 * p.grad

        total_loss += loss.data

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

# Test predictions
print("\nFinal Predictions:")
for x, _ in data:
    inputs = [Value(xi) for xi in x]
    pred = mlp(inputs)[0].data
    print(f"{x} => {pred:.4f}")
