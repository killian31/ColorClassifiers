import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from tqdm import tqdm

# Step 1: Prepare the data
data_file = "./labels.txt"

image_paths = []
color_labels = []

with open(data_file, "r") as f:
    for line in f:
        line = line.strip().split()
        image_paths.append(line[0])
        color_labels.append(line[1])

# Encode the color labels
color_encoding = {"white": 0, "yellow": 1, "red": 2, "blue": 3, "green": 4}
encoded_labels = [color_encoding[label] for label in color_labels]

# Split the dataset into training and testing sets
train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, encoded_labels, test_size=0.33, random_state=42
)


# Step 2: Load and preprocess the images
class ColorDataset(Dataset):
    def __init__(self, image_paths, color_labels, transform=None):
        self.image_paths = image_paths
        self.color_labels = color_labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        color_label = self.color_labels[index]

        if self.transform is not None:
            image = self.transform(image)

        color_label = torch.tensor(color_label).to(device)

        return image, color_label


# Define image transformations
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)

train_dataset = ColorDataset(train_paths, train_labels, transform=transform)
test_dataset = ColorDataset(test_paths, test_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Step 3: Encode the color labels
# Implement color label encoding based on your specific requirements


# Step 4: Build the neural network model
class ColorNet(nn.Module):
    def __init__(self, num_classes=5):
        super(ColorNet, self).__init__()
        # Define your model architecture
        # Specify appropriate layers and activation functions
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 64 * 64, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(
            self.conv1(x)
        )  # Apply the first convolutional layer and ReLU activation
        x = F.max_pool2d(x, 2)  # Apply max pooling
        x = F.relu(
            self.conv2(x)
        )  # Apply the second convolutional layer and ReLU activation
        x = F.max_pool2d(x, 2)  # Apply max pooling
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))  # Apply the fully connected layer and ReLU activation
        x = self.fc2(x)  # Apply the output layer

        return x


model = ColorNet()

if os.path.exists("color_weights.pth"):
    model.load_state_dict(torch.load("color_weights.pth"))
# Step 5: Train the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 9

pbar = tqdm(range(num_epochs))
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_accuracy = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        train_accuracy += (predicted == labels).sum().item()

    train_loss = train_loss / len(train_loader.dataset)
    train_accuracy = 100.0 * train_accuracy / len(train_loader.dataset)
    pbar.update(1)
    pbar.set_description(
        f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%"
    )
pbar.close()

# torch.save(model.state_dict(), "color_weights.pth")

# Step 6: Evaluate the model
model.eval()
test_loss = 0.0
test_accuracy = 0.0

for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)
    loss = criterion(outputs, labels)

    test_loss += loss.item() * images.size(0)
    _, predicted = torch.max(outputs, 1)
    test_accuracy += (predicted == labels).sum().item()

test_loss = test_loss / len(test_loader.dataset)
test_accuracy = 100.0 * test_accuracy / len(test_loader.dataset)

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
