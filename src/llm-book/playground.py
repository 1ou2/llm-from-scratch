import torch
import tiktoken
import torch.nn as nn
import matplotlib.pyplot as plt
from chapter3_gpt import GELU

def gelurelu():
    gelu, relu = GELU(), nn.ReLU()
    x = torch.linspace(-5, 5, 100)
    y_gelu = gelu(x)
    y_relu = relu(x)
    y_softmax = torch.softmax(x, dim=-1)
    plt.figure(figsize=(8, 6))
    plt.plot(x, y_gelu, label='GELU')
    #plt.plot(x, y_relu, label='ReLU')
    plt.legend()  # Add this to show the labels
    plt.tight_layout()
    plt.savefig('gelu.png', bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print("Plot 1 saved as gelu.png")

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot GELU on the left subplot
    ax1.plot(x, y_gelu, 'b-')
    ax1.set_title('GELU activation function')
    ax1.grid(True)
    
    # Plot ReLU on the right subplot
    ax2.plot(x, y_relu, 'r-')
    ax2.set_title('ReLU activation function')
    ax2.grid(True)
    
    plt.tight_layout()  # Adjust the spacing between subplots
    plt.savefig('activation_functions2x.png', bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print("Plot 2 saved as activation_functions2x.png")

    plt.figure(figsize=(8, 6))
    plt.plot(x, y_softmax, label='Softmax')
    
    plt.legend()  # Add this to show the labels
    plt.tight_layout()
    plt.savefig('softmax.png', bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print("Plot 1 saved as softmax.png")
def plot_activation_functions():
    x = torch.linspace(-5, 5, 100)
    
    # Define activation functions
    gelu = GELU()
    relu = nn.ReLU()
    leaky_relu = nn.LeakyReLU(negative_slope=0.01)
    
    # Calculate outputs
    y_gelu = gelu(x)
    y_relu = relu(x)
    y_sigmoid = torch.sigmoid(x)
    y_tanh = torch.tanh(x)
    y_leaky_relu = leaky_relu(x)
    
    # Create subplot grid
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot each activation function
    ax1.plot(x, y_gelu, 'b-')
    ax1.set_title('GELU')
    ax1.grid(True)
    
    ax2.plot(x, y_relu, 'r-')
    ax2.set_title('ReLU')
    ax2.grid(True)
    
    ax3.plot(x, y_sigmoid, 'g-')
    ax3.set_title('Sigmoid')
    ax3.grid(True)
    
    ax4.plot(x, y_tanh, 'c-')
    ax4.set_title('Tanh')
    ax4.grid(True)
    
    ax5.plot(x, y_leaky_relu, 'm-')
    ax5.set_title('Leaky ReLU')
    ax5.grid(True)
    
    # Keep one subplot empty or use it for something else
    ax6.set_visible(False)
    
    plt.tight_layout()
    plt.savefig('all_activation_functions.png', bbox_inches='tight')
    plt.close()

def matrix():
    # Example with context_len = 4
    context_len = 4

    # Create a matrix of ones
    ones = torch.ones(context_len, context_len)
    print("Matrix of ones:")
    print(ones)

    # Apply torch.triu with diagonal=1
    mask = torch.triu(ones, diagonal=1)
    print("\nUpper triangular mask (diagonal=1):")
    print(mask)

def layer_norm():

    torch.manual_seed(123)
    batch = torch.randn(2,5)
    print(f"{batch=}")
    layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    out = layer(batch)
    print(f"{out=}")
    print(f"{out.shape=}")
    mean = out.mean(dim=-1,keepdim=True)
    var = out.var(dim=-1, keepdim=True)
    print(f"{mean=}")
    print(f"{var=}")

if __name__ == "__main__":
    plot_activation_functions()
    print("done")