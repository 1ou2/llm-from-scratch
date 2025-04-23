import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle
import os
from torchviz import make_dot
import networkx as nx

def sample_net():
    inputs = torch.tensor([[0.1, -0.2, 1.3,0.11], [-0.4, 0.5, -0.6, 0.7]])

    layer1 = nn.Linear(4,5)
    layer2 = nn.Linear(5,10)
    layer3 = nn.Linear(10, 1)
    reLu = nn.ReLU()

    model = nn.Sequential(layer1, reLu, layer2, reLu, layer3)

    output = model(inputs)
    print(output)




def generate_synthetic_house_data(n_samples=1000, random_seed=42):
    """
    Generate synthetic house data with the following features:
    - distance to city center (km)
    - area (square feet)
    - number of bedrooms
    - age (years)
    """
    np.random.seed(random_seed)
    
    # Generate raw features
    distance = np.random.uniform(0, 30, n_samples)  # Distance: 0 to 30 km
    area = np.random.uniform(800, 5000, n_samples)  # Area: 800 to 5000 sq ft
    bedrooms = np.random.randint(1, 7, n_samples)   # Bedrooms: 1 to 6
    age = np.random.uniform(0, 150, n_samples)      # Age: 0 to 150 years
    
    # Create base price (in thousands of dollars)
    base_price = (
        300  # Base price $300k
        - distance * 5  # Price decreases with distance
        + area * 0.1    # Price increases with area
        + bedrooms * 50  # Each bedroom adds value
    )
    
    # Stronger age effect
    # First decreasing until ~50 years, then increasing for historic homes
    age_effect = (-0.7 * age  # Stronger initial decrease
                 + 0.01 * (age ** 2)  # Stronger quadratic term
                 - 0.00001 * (age ** 3))  # Add cubic term for more control
    
    # Combine all effects and add some random noise
    price = base_price + age_effect + np.random.normal(0, 50, n_samples)
    
    # Ensure all prices are positive
    price = np.maximum(price, 50)  # Minimum price of $50k
    
    # Create DataFrame with raw data
    df = pd.DataFrame({
        'distance': distance,
        'area': area,
        'bedrooms': bedrooms,
        'age': age,
        'price': price
    })
    
    # Create feature matrix X and target vector y
    X = df[['distance', 'area', 'bedrooms', 'age']].values
    y = df[['price']].values
    
    # Initialize scalers
    feature_scaler = StandardScaler()
    price_scaler = StandardScaler()
    
    # Scale features and target
    X_scaled = feature_scaler.fit_transform(X)
    y_scaled = price_scaler.fit_transform(y)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y_scaled)
    
    return X_tensor, y_tensor, feature_scaler, price_scaler

def visualize_synthetic_data():
    # Generate data
    df_raw, X_train, y_train, scaler = generate_synthetic_house_data()

    # Visualize the relationships
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()

    axes[0].scatter(df_raw['distance'], df_raw['price'], alpha=0.5)
    axes[0].set_xlabel('Distance to City Center (km)')
    axes[0].set_ylabel('Price ($k)')
    axes[0].set_title('Price vs Distance')

    axes[1].scatter(df_raw['area'], df_raw['price'], alpha=0.5)
    axes[1].set_xlabel('Area (sq ft)')
    axes[1].set_ylabel('Price ($k)')
    axes[1].set_title('Price vs Area')

    axes[2].scatter(df_raw['bedrooms'], df_raw['price'], alpha=0.5)
    axes[2].set_xlabel('Number of Bedrooms')
    axes[2].set_ylabel('Price ($k)')
    axes[2].set_title('Price vs Bedrooms')

    axes[3].scatter(df_raw['age'], df_raw['price'], alpha=0.5)
    axes[3].set_xlabel('Age (years)')
    axes[3].set_ylabel('Price ($k)')
    axes[3].set_title('Price vs Age')

    plt.tight_layout()
    plt.savefig("prices_dataset.png")

    # Print some sample data
    print("\nSample of raw data:")
    print(df_raw.head())
    print("\nShape of training tensors:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")


class SimpleHouseNet(nn.Module):
    def __init__(self):
        super(SimpleHouseNet, self).__init__()
        # Input layer: 4 features (distance, area, bedrooms, age)
        # Hidden layer: 8 neurons 
        # Output layer: 1 (price)
        self.layer1 = nn.Linear(4, 8)
        self.layer2 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x
    
# Modified network with one more layer
class House3Layers(nn.Module):
    def __init__(self):
        super(House3Layers, self).__init__()
        self.layer1 = nn.Linear(4, 5)  # Increased first layer
        self.layer2 = nn.Linear(5, 4)  # Added middle layer
        self.layer3 = nn.Linear(4, 1)   # Output layer
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

def train_model(model,learning_rate,n_epochs,X_train,y_train):
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    losses = []
    for epoch in range(n_epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Store loss
        losses.append(loss.item())
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

    # Plot training curve
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.savefig("houses_training_loss.png")

    return model, losses


def save_model(model, feature_scaler, price_scaler, folder_path='saved_model',prefix=""):
    """
    Save the trained model, raw data, and scalers
    """
    # Create folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Save the model
    torch.save(model.state_dict(), os.path.join(folder_path, prefix+'house_model.pth'))

    # Save the scalers
    with open(os.path.join(folder_path, prefix+ 'scalers.pkl'), 'wb') as f:
        pickle.dump({
            'feature_scaler': feature_scaler,
            'price_scaler': price_scaler
        }, f)
    
    print(f"Model and data saved in folder: {folder_path}")

def load_model(model_class, folder_path='saved_model', prefix=""):
    """
    Load the trained model and scalers
    
    Args:
        model_class: The model class (SimpleHouseNet or SimpleHouseNet2)
        folder_path: Path to the folder containing saved model files
        prefix: Prefix used when saving the model (e.g., "trained" or "untrained")
    
    Returns:
        model: The loaded PyTorch model
        feature_scaler: The loaded feature scaler
        price_scaler: The loaded price scaler
    """
    # Initialize a new model instance
    model = model_class()
    
    # Load the model state dictionary
    model_path = os.path.join(folder_path, prefix + 'house_model.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    
    # Load the scalers
    scalers_path = os.path.join(folder_path, prefix + 'scalers.pkl')
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
        
    feature_scaler = scalers['feature_scaler']
    price_scaler = scalers['price_scaler']
    
    return model, feature_scaler, price_scaler


def save_data(X,y,folder_path="saved_model"):
    # Y and y contains dataframe values
    # X = df[['distance', 'area', 'bedrooms', 'age']].values
    # y = df[['price']].values
    # save to file the synthetic data generated
    # Save both arrays in a single compressed file
    np.savez_compressed(os.path.join(folder_path, "training_data.npz"), X=X, y=y)
    
def load_data(file_path="saved_model/training_data.npz"):
    data = np.load(file_path)
    X = data['X']
    y = data['y']
    return X, y


# Test the model with some sample data
def predict_price(model, distance, area, bedrooms, age, feature_scaler, price_scaler):
    # Create input array
    X_test = np.array([[distance, area, bedrooms, age]])
    
    # Scale input
    X_test_scaled = feature_scaler.transform(X_test)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor)
        # Reshape prediction to 2D array for inverse transform
        y_pred_scaled_np = y_pred_scaled.numpy().reshape(-1, 1)
        # Inverse transform to get actual price
        y_pred = price_scaler.inverse_transform(y_pred_scaled_np)
    
    return y_pred[0][0]

def run_test_cases(model,feature_scaler,price_scaler):
        # Example predictions
    test_cases = [
        (1, 4000, 6, 1),    # New house close to city
        (1, 3000, 6, 1),    # New house close to city
        (10, 8000, 10, 5),    # New house close to city
        (20, 8000, 10, 5),    # New house close to city
        (20, 2000, 3, 5),   # older house far from city
        (20, 2000, 3, 8),   # even older
        (20, 2000, 3, 100),    # very old house
    ]
    print("\nExample predictions:")
    for distance, area, bedrooms, age in test_cases:
        price = predict_price(model,distance, area, bedrooms, age, feature_scaler,price_scaler)
        print(f"House: {distance}km from city,area {area}sqft, {bedrooms} beds, {age} years old")
        print(f"Predicted price: ${price:.2f}k\n")

def pretraining():
    # Create model
    model = SimpleHouseNet()

    # Example usage with our previously generated data
    print("Model architecture:")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal number of parameters: {total_params}")



        # Generate data
    X_train, y_train,feature_scaler, price_scaler = generate_synthetic_house_data(n_samples=10000)
    save_data(X_train, y_train)

    run_test_cases(model,feature_scaler,price_scaler)
    save_model(model,feature_scaler,price_scaler,prefix="untrained")

    learning_rate = 0.001  # Good starting point for Adam optimizer
    n_epochs = 1000
    
    # Train model
    model, losses = train_model(model, learning_rate, n_epochs, X_train, y_train)
    run_test_cases(model,feature_scaler,price_scaler)
    save_model(model,feature_scaler,price_scaler,prefix="trained")

def visualize_network(model):
    # Create graph
    G = nx.DiGraph()
    
    # Add input nodes
    input_features = ['Distance', 'Area', 'Bedrooms', 'Age']
    for i in range(4):
        G.add_node(f'i{i}', pos=(0, i), layer='input', label=input_features[i])
    
    # Add hidden layer nodes
    for i in range(8):
        G.add_node(f'h{i}', pos=(1, i-1.5), layer='hidden', label=f'H{i}')
    
    # Add output node
    G.add_node('o0', pos=(2, 2), layer='output', label='Price')
    
    # Add edges
    # Input to hidden
    for i in range(4):
        for h in range(8):
            G.add_edge(f'i{i}', f'h{h}')
    
    # Hidden to output
    for h in range(8):
        G.add_edge(f'h{h}', 'o0')
    
    # Draw the network
    plt.figure(figsize=(12, 8))
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                          node_color=['lightblue' if layer=='input' else 'lightgreen' if layer=='hidden' else 'lightpink' 
                                    for layer in nx.get_node_attributes(G, 'layer').values()],
                          node_size=1000)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos)
    
    # Add labels
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels)
    
    plt.title('Neural Network Architecture')
    plt.axis('off')
    plt.savefig("houses_network.png")


def print_model_params(model):
    # First layer (4 inputs -> 8 neurons)
    print("Layer 1:")
    print("Weights shape:", model.layer1.weight.shape)
    print("Weights:\n", model.layer1.weight.data)
    print("Biases shape:", model.layer1.bias.shape)
    print("Biases:\n", model.layer1.bias.data)
    
    print("\nLayer 2:")
    # Second layer (8 neurons -> 1 output)
    print("Weights shape:", model.layer2.weight.shape)
    print("Weights:\n", model.layer2.weight.data)
    print("Biases shape:", model.layer2.bias.shape)
    print("Biases:\n", model.layer2.bias.data)





if __name__ == "__main__":
    #visualize_synthetic_data()
    #pretraining()

    # Example usage
    try:
        # Load the trained model
        model, feature_scaler, price_scaler = load_model(
            model_class=SimpleHouseNet,  # or SimpleHouseNet2
            folder_path='saved_model',
            prefix="trained"  # or "untrained"
        )
        #run_test_cases(model, feature_scaler, price_scaler)
        # Now you can use the model for predictions
        # Example test case
        distance, area, bedrooms, age = 5, 2000, 3, 15
        price = predict_price(model, distance, area, bedrooms, age, 
                             feature_scaler, price_scaler)
        #print(f"Predicted price: ${price:.2f}k")

        # Usage example:
        
        print_model_params(model)
        model = SimpleHouseNet()
        print_model_params(model)
        
    except FileNotFoundError:
        print("Error: Model files not found. Please ensure the model has been saved first.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

    # Call the visualization
    #visualize_network(model)
