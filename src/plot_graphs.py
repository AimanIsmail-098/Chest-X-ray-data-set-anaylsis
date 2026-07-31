import matplotlib.pyplot as plt

# Training History
train_accuracy = [
    0.8886119723320007, 0.9562883377075195, 0.9722009301185608,
    0.9773772954940796, 0.9787193536758423, 0.9850460290908813,
    0.9869632124900818, 0.994056761264801, 0.9948236346244812,
    0.9965490698814392
]

val_accuracy = [
    0.6875, 0.8125, 0.9375, 0.625, 0.875,
    0.875, 0.75, 0.875, 0.875, 1.0
]

train_loss = [
    0.2851293683052063, 0.12073183059692383, 0.08473440259695053,
    0.06320835649967194, 0.058324094861745834, 0.04216586798429489,
    0.030539214611053467, 0.01791362836956978,
    0.012627901509404182, 0.00985609833151102
]

val_loss = [
    0.6697049140930176, 0.36231356859207153, 0.16663479804992676,
    1.3834210634231567, 0.3472438156604767, 0.3171123266220093,
    0.6511229872703552, 0.24354498088359833,
    0.13596124947071075, 0.0174920205026865
]

epochs = range(1, len(train_accuracy) + 1)

# ---------------- Accuracy Curve ----------------
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_accuracy, marker='o', label='Training Accuracy')
plt.plot(epochs, val_accuracy, marker='s', label='Validation Accuracy')

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xticks(epochs)
plt.grid(True)
plt.legend()

plt.savefig("accuracy_curve.png")
plt.show()

# ---------------- Loss Curve ----------------
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, marker='o', label='Training Loss')
plt.plot(epochs, val_loss, marker='s', label='Validation Loss')

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(epochs)
plt.grid(True)
plt.legend()

plt.savefig("loss_curve.png")
plt.show()

print("Graphs saved successfully!")