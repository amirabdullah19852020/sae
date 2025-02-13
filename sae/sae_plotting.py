from math import ceil

import json
import plotly.graph_objects as go

from datasets import load_dataset
from nnsight import LanguageModel
from plotly.subplots import make_subplots

from .sae import Sae

def plot_layer_curves(layer_data, cols=2):
    """
    Plot cumulative error against features included for each layer in a grid layout.

    Parameters:
    layer_data (dict): Dictionary mapping layer names to dictionaries of {features_included: cumulative_error}.
                       Example:
                       {
                           "Layer A": {100: 0, 80: 0.1, 60: 0.3, 40: 0.6, 20: 0.8, 0: 1},
                           "Layer B": {120: 0, 100: 0.05, 80: 0.15, 60: 0.4, 40: 0.65, 20: 0.85, 0: 1}
                       }
    cols (int): Number of columns in the grid layout. Default is 2.
    """
    # Calculate number of layers and rows required
    num_layers = len(layer_data)
    rows = ceil(num_layers / cols)

    # Create a subplot figure
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=list(layer_data.keys()),
        horizontal_spacing=0.1, vertical_spacing=0.2
    )

    # Add traces for each layer
    for i, (layer_name, data) in enumerate(layer_data.items()):
        # Sort features to ensure the order is descending by keys (features included)
        sorted_data = sorted(data.items(), key=lambda x: x[0], reverse=True)
        features, errors = zip(*sorted_data)

        # Determine the row and column for the subplot
        row = i // cols + 1
        col = i % cols + 1

        # Add a line trace for the current layer
        fig.add_trace(
            go.Scatter(
                x=features, y=errors,
                mode='lines+markers',
                name=layer_name,
                line=dict(width=2),
                marker=dict(size=6)
            ),
            row=row, col=col
        )

    # Update layout
    fig.update_layout(
        height=300 * rows,  # Adjust height based on number of rows
        width=800,  # Fixed width
        title_text="Cumulative Error vs Features Included",
        title_font_size=24,
        title_x=0.5,
        showlegend=False
    )
    fig.update_xaxes(title_text="Features Included")  # Reverse x-axis for descending features
    fig.update_yaxes(title_text="Cumulative Error")

    # Show the plot
    fig.show()

    fig.write_image("reconstruction_loss_curves.png")


def plot_layer_features(full_layer_data, tag, model_name, top_n=20, cols=2):
    """
    Plot histograms for the top `n` features in each layer using a Plotly grid.

    Parameters:
    layer_data (dict): Dictionary where keys are layer names and values are lists of tuples
                       (feature_num, weight), sorted in descending order by weight.
    model_name (str): Model name
    top_n (int): Number of top features to display for each layer.
    cols (int): Number of columns in the grid layout.
    """
    # Calculate rows based on number of layers and columns

    layer_data = {layer: full_layer_data[layer]["sorted_weights"] for layer, layer_data in full_layer_data.items()}

    num_layers = len(layer_data)
    rows = ceil(num_layers / cols)

    # Create a Plotly figure with subplots
    fig = make_subplots(
        rows=rows, cols=cols, subplot_titles=list(layer_data.keys()),
        horizontal_spacing=0.1, vertical_spacing=0.2
    )

    # Add data to subplots
    for i, (layer_name, features) in enumerate(layer_data.items()):
        top_features = features[:top_n]
        feature_labels, weights = zip(*top_features)
        feature_labels = [str(label) for label in feature_labels]

        # Determine the row and column for the subplot
        row = i // cols + 1
        col = i % cols + 1

        # Add a bar chart for the layer
        fig.add_trace(
            go.Bar(x=feature_labels, y=weights, name=layer_name),
            row=row, col=col
        )

    # Update layout
    fig.update_layout(
        height=300 * rows,  # Adjust height based on rows
        title_text=f"Top Features by Layer for {tag}",
        title_x=0.5,
        showlegend=False,
    )
    fig.update_xaxes(title_text="Feature Number", tickangle=90)
    fig.update_yaxes(title_text="Weight")

    fig.write_image(f"SAE_top_features_for_{tag}_{model_name}.png")

    # Show the plot
    fig.show()


def visualize_tensor_blocks(tensor, block_size, output_file):
    """
    Visualizes the magnitudes of contiguous blocks of a tensor using a Plotly histogram
    and saves the plot as a PNG file.

    Args:
        tensor (torch.Tensor): The input tensor.
        block_size (int): The size of each block.
        output_file (str): The file path to save the PNG image.
    """
    if tensor.numel() % block_size != 0:
        raise ValueError("Tensor size must be divisible by the block size.")

    # Split the tensor into contiguous blocks
    num_blocks = tensor.numel() // block_size
    blocks = tensor.split(block_size)

    # Calculate the magnitude (sum of absolute values) for each block
    magnitudes = [block.abs().sum().item() for block in blocks]

    # Create the Plotly bar plot
    fig = go.Figure(data=[
        go.Bar(x=list(range(num_blocks)), y=magnitudes, marker_color='skyblue')
    ])

    feature_name = output_file.replace(".png", "")

    fig.update_layout(
        title=f"Attention head outputs for {feature_name}",
        xaxis_title="Block Index",
        yaxis_title="Magnitude (Sum of Absolute Values)",
        xaxis=dict(tickmode='linear'),
        template="plotly_white"
    )

    # Save the plot to a file
    fig.write_image(output_file)
    print(f"Plot saved to {output_file}")

    fig.show()
    return magnitudes

def plot_layer_curves(layer_data, cols=2):
    """
    Plot cumulative error against features included for each layer in a grid layout.

    Parameters:
    layer_data (dict): Dictionary mapping layer names to dictionaries of {features_included: cumulative_error}.
                       Example:
                       {
                           "Layer A": {100: 0, 80: 0.1, 60: 0.3, 40: 0.6, 20: 0.8, 0: 1},
                           "Layer B": {120: 0, 100: 0.05, 80: 0.15, 60: 0.4, 40: 0.65, 20: 0.85, 0: 1}
                       }
    cols (int): Number of columns in the grid layout. Default is 2.
    """
    # Calculate number of layers and rows required
    num_layers = len(layer_data)
    rows = ceil(num_layers / cols)

    # Create a subplot figure
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=list(layer_data.keys()),
        horizontal_spacing=0.1, vertical_spacing=0.2
    )

    # Add traces for each layer
    for i, (layer_name, data) in enumerate(layer_data.items()):
        # Sort features to ensure the order is descending by keys (features included)
        sorted_data = sorted(data.items(), key=lambda x: x[0], reverse=True)
        features, errors = zip(*sorted_data)

        # Determine the row and column for the subplot
        row = i // cols + 1
        col = i % cols + 1

        # Add a line trace for the current layer
        fig.add_trace(
            go.Scatter(
                x=features, y=errors,
                mode='lines+markers',
                name=layer_name,
                line=dict(width=2),
                marker=dict(size=6)
            ),
            row=row, col=col
        )

    # Update layout
    fig.update_layout(
        height=300 * rows,  # Adjust height based on number of rows
        width=800,  # Fixed width
        title_text="Cumulative Error vs Features Included",
        title_font_size=24,
        title_x=0.5,
        showlegend=False
    )
    fig.update_xaxes(title_text="Features Included")  # Reverse x-axis for descending features
    fig.update_yaxes(title_text="Cumulative Error")

    # Show the plot
    fig.show()

    fig.write_image("reconstruction_loss_curves.png")
