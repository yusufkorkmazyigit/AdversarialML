import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

print("Working directory:", os.getcwd())
print("MNIST raw exists:",
      os.path.exists("./data/MNIST/raw"))
print("Files:",
      os.listdir("./data/MNIST/raw") if os.path.exists("./data/MNIST/raw") else "YOK")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=True
)

print("Train size:", len(train_dataset))
print("Test size:", len(test_dataset))


model = SimpleNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

epochs = 3  # sunum için yeterli

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}")


model.eval()

data, target = next(iter(test_loader))
data, target = data.to(device), target.to(device)

output = model(data)
pred = output.argmax(dim=1)

print("Gerçek etiket:", target.item())
print("Model tahmini:", pred.item())

plt.imshow(data.detach().cpu().squeeze(), cmap="gray")
plt.title(f"Normal prediction: {pred.item()}")
plt.axis("off")
plt.show()


# FGSM saldırısı

# ================== FGSM ATTACK ==================

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


epsilon = 0.25

data.requires_grad = True
output = model(data)
loss = criterion(output, target)

model.zero_grad()
loss.backward()

data_grad = data.grad.data
perturbed_data = fgsm_attack(data, epsilon, data_grad)

output_adv = model(perturbed_data)
pred_adv = output_adv.argmax(dim=1)

# --- FGSM attack sonrası ---

print("Normal tahmin:", pred.item())
print("Adversarial tahmin:", pred_adv.item())


# Orijinal ve adversarial görüntü
plt.figure(figsize=(6,3))

plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(data.detach().cpu().squeeze(), cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Adversarial Image")
plt.imshow(perturbed_data.detach().cpu().squeeze(), cmap="gray")
plt.axis("off")

plt.show()

# --- BONUS: perturbation (noise) ---
noise = perturbed_data - data

plt.figure(figsize=(3,3))
plt.title("Adversarial Noise (ε = {})".format(epsilon))
plt.imshow(noise.detach().cpu().squeeze(), cmap="gray")
plt.colorbar()
plt.axis("off")
plt.show()

