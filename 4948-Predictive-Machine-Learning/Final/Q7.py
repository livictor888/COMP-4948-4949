import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_csv("C:\datasets\wildlife.csv")

# Split the dataset into features (X) and target (y)
X = df.drop('species', axis=1).values
y = df['species'].values

# Encode the target labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

input_dim = X_train.shape[1]
output_dim = len(np.unique(y_train))

model = Net(input_dim, output_dim)

# Set up the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
losses = []

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# Evaluate the model
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_prob = torch.softmax(y_pred, dim=1)
    y_pred_label = torch.argmax(y_pred_prob, dim=1)
    accuracy = accuracy_score(y_test, y_pred_label)
    precision = precision_score(y_test, y_pred_label, average='weighted')
    recall = recall_score(y_test, y_pred_label, average='weighted')
    f1 = f1_score(y_test, y_pred_label, average='weighted')

print("Accuracy on Test Set: {:.2f}".format(accuracy))
print("Precision on Test Set: {:.2f}".format(precision))
print("Recall on Test Set: {:.2f}".format(recall))
print("F1 Score on Test Set: {:.2f}".format(f1))

# Visualize progress over multiple epochs
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Learning Progress')
plt.show()
