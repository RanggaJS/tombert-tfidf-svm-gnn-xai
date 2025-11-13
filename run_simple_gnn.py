# Simple GNN test without sparse matrix issues
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Simple MLP for GNN testing
class SimpleGNN(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=256, output_dim=3):
        super(SimpleGNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Load data
def load_data():
    import pickle
    with open('./absa_data/twitter2015/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

# Train simple model
data = load_data()
print("Data loaded successfully!")

# Create simple features
from sklearn.feature_extraction.text import TfidfVectorizer

# Prepare text features
train_texts = [f"text_{i}" for i in range(len(data['X_train']))]
test_texts = [f"text_{i}" for i in range(len(data['X_test']))]

tfidf = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
train_features = tfidf.fit_transform(train_texts).toarray()
test_features = tfidf.transform(test_texts).toarray()

# Convert to tensors
device = torch.device('cpu')
train_features = torch.FloatTensor(train_features).to(device)
train_labels = torch.LongTensor(data['y_train'].numpy()).to(device)
test_features = torch.FloatTensor(test_features).to(device)
test_labels = torch.LongTensor(data['y_test'].numpy()).to(device)

# Train model
model = SimpleGNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_features)
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    test_outputs = model(test_features)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = accuracy_score(test_labels.cpu().numpy(), predicted.cpu().numpy())
    report = classification_report(test_labels.cpu().numpy(), predicted.cpu().numpy(),
                                  target_names=['Negative', 'Neutral', 'Positive'])
    
print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"\nClassification Report:\n{report}")
