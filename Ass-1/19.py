import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Device configuration (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transformations for the data (normalize and convert to tensor)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and load FashionMNIST dataset
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=1000, shuffle=False)

# Define the Fully Connected Network (FCN)
class FullyConnectedNN(nn.Module):
    def __init__(self):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Flatten 28x28 images
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # 10 output classes (FashionMNIST)
    
    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the image
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the Convolutional Neural Network (CNN)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.pool = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        self.fc1 = nn.Linear(64 * 14 * 14, 128)  # Flatten 14x14 images
        self.fc2 = nn.Linear(128, 10)  # 10 output classes (FashionMNIST)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 14 * 14)  # Flatten the image
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the models
fcn_model = FullyConnectedNN().to(device)
cnn_model = CNN().to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizers (SGD with momentum)
optimizer_fcn = optim.SGD(fcn_model.parameters(), lr=0.01, momentum=0.9)
optimizer_cnn = optim.SGD(cnn_model.parameters(), lr=0.01, momentum=0.9)

# Function to train a model
def train_model(model, trainloader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(trainloader)}")
    print("Finished Training")

# Train the models
print("Training Fully Connected Network:")
train_model(fcn_model, trainloader, optimizer_fcn, criterion)

print("\nTraining Convolutional Neural Network:")
train_model(cnn_model, trainloader, optimizer_cnn, criterion)

# Function to evaluate a model
def evaluate_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# Evaluate the models
print("\nEvaluating Fully Connected Network:")
fcn_test_accuracy = evaluate_model(fcn_model, testloader)

print("\nEvaluating Convolutional Neural Network:")
cnn_test_accuracy = evaluate_model(cnn_model, testloader)

# Plot comparison of test accuracy
plt.bar(['Fully Connected Network', 'CNN'], [fcn_test_accuracy, cnn_test_accuracy])
plt.ylabel('Test Accuracy (%)')
plt.title('Test Accuracy Comparison')
plt.show()
