from torch_neuralnet import SimpleHouseNet, load_model
import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from torch import nn

# RUN
# ~/dev/llm-from-scratch$ streamlit run src/training/backprop/vizneuralnet.py 
#
# Visualize a toy neural network used to predict house prices

def get_layer_activations(model, input_tensor):
    """Get activations from hidden layers"""
    activations = []
    
    def hook(model, input, output):
        activations.append(output.detach().numpy())
    
    # Register hooks for each linear layer
    hooks = []
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            hooks.append(layer.register_forward_hook(hook))
    
    # Forward pass
    _ = model(input_tensor)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return activations

def create_network_visualization(activations, input_values):
    """Create a visualization of the network with activation values"""
    # Define colors based on activation values
    def get_color(value):
        # Normalize value using sigmoid or clip between 0 and 1
        normalized = np.clip(value / 5.0, 0, 1)
        # Convert to color: yellow (low activation) to red (high activation)
        return f'rgb(255, {int(255*max(1-normalized, normalized))}, 0)'
    
    # Create the visualization
    fig = go.Figure()
    
    # Add nodes for input layer
    layer_x = [0] * 4
    layer_y = [-1.5, -0.5, 0.5, 1.5]
    input_labels = ['Distance', 'Area', 'Bedrooms', 'Age']
    
    for i, (y, label, value) in enumerate(zip(layer_y, input_labels, input_values)):
        fig.add_trace(go.Scatter(
            x=[0], y=[y],
            mode='markers+text',
            marker=dict(size=40, color='lightgray',line=dict(color='black', width=2)),
            text=[f'{label}<br>{value:.2f}'],
            textposition="middle right",
            textfont=dict(size=14),
            name=f'Input {label}'
        ))
    
    # Add nodes for hidden layers
    for layer_idx, layer_activations in enumerate(activations[:-1]):  # Exclude output layer
        layer_x = [(layer_idx + 1) * 2] * len(layer_activations[0])
        layer_y = np.linspace(-2, 2, len(layer_activations[0]))
        
        for i, activation in enumerate(layer_activations[0]):
            fig.add_trace(go.Scatter(
                x=[layer_x[i]], y=[layer_y[i]],
                mode='markers+text',
                marker=dict(
                    size=40,  # Increased node size
                    color=get_color(activation),
                    line=dict(color='black', width=2)
                ),
                text=[f'{activation:.2f}'],
                textfont=dict(
                    size=14,  # Increased font size
                    color='black'  # Changed text color to black
                ),
                name=f'Hidden {layer_idx+1}-{i+1}'
            ))
    
    # Add output node
    fig.add_trace(go.Scatter(
        x=[len(activations) * 2], y=[0],
        mode='markers+text',
        marker=dict(
            size=40,  # Increased node size
            color='lightgray',  # Changed to black for better contrast
            line=dict(color='black', width=2)
        ),
        text=[f'Price<br>{activations[-1][0][0]:.2f}'],
        textposition="middle left",
        textfont=dict(size=14),  # Increased font size
        name='Output'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="Neural Network Visualization",
            font=dict(size=24)  # Increased title font size
        ),
        showlegend=False,
        plot_bgcolor='white',
        height=600,  # Increased height
        width=1000,  # Increased width
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig

def main():
    st.title("House Price Prediction")
    st.write("Enter house details to predict price")
    
    # Input fields
    col1, col2 = st.columns(2)
    with col1:
        distance = st.number_input("Distance from city center (km)", min_value=0.0, value=1.0)
        area = st.number_input("Area (sq ft)", min_value=0.0, value=2000.0)
    with col2:
        bedrooms = st.number_input("Number of bedrooms", min_value=0, value=3)
        age = st.number_input("Age of house (years)", min_value=0, value=5)
    
    # Load model and scalers
    try:
        model, feature_scaler, price_scaler = load_model(
            model_class=SimpleHouseNet,
            folder_path='saved_model',
            prefix="trained"
        )
        
        if st.button("Predict Price"):
            # Prepare input
            input_data = np.array([[distance, area, bedrooms, age]])
            scaled_input = feature_scaler.transform(input_data)
            input_tensor = torch.FloatTensor(scaled_input)
            
            # Get activations and prediction
            activations = get_layer_activations(model, input_tensor)
            
            # Get prediction
            with torch.no_grad():
                scaled_prediction = model(input_tensor)
                prediction = price_scaler.inverse_transform(scaled_prediction.numpy())[0][0]
            
            # Display prediction
            st.subheader(f"Predicted Price: {prediction:,.2f} k$")
            
            # Display network visualization
            fig = create_network_visualization(activations, scaled_input[0])
            st.plotly_chart(fig)
            
            # Additional information
            st.write("### Network Interpretation")
            st.write("- Redder nodes indicate stronger positive activations")
            st.write("- Yellow nodes indicate stronger negative activations")
            st.write("- The intensity of the color represents the strength of the activation")
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")

if __name__ == "__main__":
    main()
