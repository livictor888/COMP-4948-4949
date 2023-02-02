import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split

### DATA           #######################################
# Setup data.
import pandas as pd
import numpy as np
candidates = {'gmat': [780,750,690,710,680,730,690,720,
 740,690,610,690,710,680,770,610,580,650,540,590,620,
 600,550,550,570,670,660,580,650,660,640,620,660,660,
 680,650,670,580,590,690],
              'gpa': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,
 3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,
 3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,
 3.3,3.3,2.3,2.7,3.3,1.7,3.7],
              'work_experience': [3,4,3,5,4,6,1,4,5,
 1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,
 5,1,2,1,4,5],
              'admitted': [1,1,1,1,1,1,0,1,1,0,0,1,
 1,1,1,0,0,1,0,0,0,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,
 0,0,1]}

df = pd.DataFrame(candidates,columns= ['gmat', 'gpa',
                  'work_experience','admitted'])
y = np.array(df['admitted'])
X = df.copy()
del X['admitted']
X = X
##########################################################
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# define standard scaler
from sklearn.preprocessing     import StandardScaler
scaler = StandardScaler()

# transform data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test  = torch.tensor(X_test_scaled, dtype=torch.float32)

# Reshapes array.
# unsqueeze() creates array of single dimensional arrays.
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test  = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

import torch
import torch.nn as nn

# Define the neural network architecture
class BinaryClassificationNet(nn.Module):
    def __init__(self):
        super(BinaryClassificationNet, self).__init__()
        self.fc1 = nn.Linear(3, 8)
        self.fc2 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)     # Hidden layer.
        x = self.sigmoid(x) # Activation function.
        x = self.fc2(x)     # Output layer.
        x = self.sigmoid(x) # Activation function.
        return x

# Instantiate the model
model = BinaryClassificationNet()

# Define the loss function and optimizer
criterion = nn.BCELoss() # Binary cross entropy.
# Use stochastic gradient descent to update weights & bias.
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
for epoch in range(2000):
    print("Epoch: " + str(epoch))
    # Forward pass
    output = model(X_train)
    loss = criterion(output, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Evaluate the model
with torch.no_grad():
    outputs = model(X_test)
    predictions = outputs.round()
    accuracy = (predictions == y_test).float().mean()
    print(f'Accuracy: {accuracy}')
