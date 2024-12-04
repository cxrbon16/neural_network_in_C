import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def parse_w_matrices(file_path):
    """
    Parse W matrices from a text file with multiple epochs
    
    Parameters:
    file_path (str): Path to the text file containing W matrices
    
    Returns:
    tuple: (W matrices, epoch information)
    """
    # Read the file
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Split content by epochs
    epoch_sections = content.split('EPOCH')
    
    # Initialize lists to store matrices and epochs
    w_matrices = []
    epochs = []
    
    # Parse each epoch section
    for section in epoch_sections:
        if not section.strip():
            continue
        
        # Split section into lines and remove empty lines
        lines = [line.strip() for line in section.split('\n') if line.strip()]
        
        # Extract epoch number
        if lines and lines[0].isdigit():
            epoch = int(lines[0])
            lines = lines[1:]
        else:
            epoch = len(w_matrices)  # Default to matrix count if no explicit epoch
        
        # Convert lines to numpy array of floats
        try:
            matrix = np.array([
                [float(x) for x in line.split()] 
                for line in lines
            ])
            w_matrices.append(matrix)
            epochs.append(epoch)
        except ValueError:
            print(f"Warning: Could not parse matrix in section: {section}")
    
    return np.array(w_matrices), epochs

def visualize_w_matrices(file_path):
    """
    Visualize W matrices using T-SNE with process-based coloring
    
    Parameters:
    file_path (str): Path to the text file containing W matrices
    
    Returns:
    Matplotlib figure with T-SNE visualization
    """
    # Parse matrices and epochs
    W_matrices, epochs = parse_w_matrices(file_path)
    
    # Reshape matrices to 2D for T-SNE
    W_reshaped = W_matrices.reshape(W_matrices.shape[0], -1)
    
    # Perform T-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    W_tsne = tsne.fit_transform(W_reshaped)
    
    # Create a color map based on unique epochs
    unique_epochs = sorted(set(epochs))
    color_map = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(unique_epochs)))
    
    # Create the figure before plotting
    plt.figure(figsize=(12, 10))
    
    # Create a scatter plot for each unique epoch
    for epoch in unique_epochs:
        # Find indices of matrices from this epoch
        epoch_indices = [i for i, e in enumerate(epochs) if e == epoch]
        
        # Plot matrices from this epoch
        plt.scatter(
            W_tsne[epoch_indices, 0], 
            W_tsne[epoch_indices, 1],
            color=color_map[unique_epochs.index(epoch)],
            label=f'Epoch {epoch}',
            s=200
        )
    
    plt.title('T-SNE Visualization of W Matrices by Epoch')
    plt.xlabel('T-SNE Dimension 1')
    plt.ylabel('T-SNE Dimension 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return plt

# Main execution
file_path = 'weight1toN.txt'

# Create the plot
plt.close('all')  # Close any existing figures
plt.ioff()  # Turn off interactive mode

# Visualize and save
plot = visualize_w_matrices(file_path)

# Save before showing
plot.savefig('w_matrices_tsne_GD.pdf', format='pdf', bbox_inches='tight', dpi=300)
plot.savefig('w_matrices_tsne.eps', format='eps', bbox_inches='tight')

# Optional: show the plot
plot.show()