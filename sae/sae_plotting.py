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


def plot_layer_features(full_layer_data, tag, top_n=20, cols=2):
    """
    Plot histograms for the top `n` features in each layer using a Plotly grid.

    Parameters:
    layer_data (dict): Dictionary where keys are layer names and values are lists of tuples
                       (feature_num, weight), sorted in descending order by weight.
    top_n (int): Number of top features to display for each layer.
    cols (int): Number of columns in the grid layout.
    """
    # Calculate rows based on number of layers and columns

    layer_data = {layer: full_layer_data[layer]["sorted_weights"] for layer, layer_data in full_layer_data.items()}

    num_layers = len(layer_data)
    rows = ceil(num_layers / cols)

    # Create a Plotly figure with subplots
    fig = make_subplots(
        rows=rows, cols=cols, x_title=f"SAE features for TAG: {tag}",
        subplot_titles=list(layer_data.keys()),
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

    fig.write_image(f"SAE_top_features_for_{tag}.png")

    # Show the plot
    fig.show()

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
