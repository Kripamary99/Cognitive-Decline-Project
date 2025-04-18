import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_radar_chart(data, features, title="Speech Features Radar Chart"):
    """Create an interactive radar chart using Plotly."""
    fig = go.Figure()
    
    # Calculate statistics
    mean_values = data[features].mean()
    std_values = data[features].std()
    
    # Add mean trace
    fig.add_trace(go.Scatterpolar(
        r=mean_values,
        theta=features,
        fill='toself',
        name='Mean'
    ))
    
    # Add ±1 std deviation traces
    fig.add_trace(go.Scatterpolar(
        r=mean_values + std_values,
        theta=features,
        fill='tonext',
        name='+1 STD'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=mean_values - std_values,
        theta=features,
        fill='tonext',
        name='-1 STD'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(mean_values + std_values)])),
        showlegend=True,
        title=title
    )
    
    fig.write_html("radar_chart.html")

def create_boxplots(data, features):
    """Create boxplots for feature distributions."""
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data[features])
    plt.xticks(rotation=45)
    plt.title("Feature Distributions")
    plt.tight_layout()
    plt.savefig("boxplots.png")
    plt.close()

def create_tsne_visualization(data, features, perplexity=30):
    """Create t-SNE visualization of the feature space."""
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_results = tsne.fit_transform(data[features])
    
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=data['risk_score'], cmap='viridis')
    plt.colorbar(label='Risk Score')
    plt.title("t-SNE Visualization of Speech Features")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.savefig("tsne_visualization.png")
    plt.close()

def plot_risk_score_density(data):
    """Create density plot for risk scores."""
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=data, x='risk_score', fill=True)
    plt.title("Risk Score Distribution")
    plt.xlabel("Risk Score")
    plt.ylabel("Density")
    plt.savefig("risk_score_density.png")
    plt.close()

def create_correlation_heatmap(data, features):
    """Create correlation heatmap for features."""
    plt.figure(figsize=(10, 8))
    correlation_matrix = data[features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    plt.close()

def generate_sample_data(n_samples=100):
    """Generate synthetic speech feature data for demonstration."""
    np.random.seed(42)
    features = {
        'pitch': np.random.normal(220, 30, n_samples),
        'intensity': np.random.normal(70, 10, n_samples),
        'jitter': np.random.normal(2, 0.5, n_samples),
        'shimmer': np.random.normal(7, 1, n_samples),
        'hnr': np.random.normal(15, 3, n_samples),
        'risk_score': np.random.uniform(0, 1, n_samples)
    }
    return pd.DataFrame(features)

def main():
    # Generate sample data
    data = generate_sample_data()
    features = ['pitch', 'intensity', 'jitter', 'shimmer', 'hnr']
    
    # Create visualizations
    create_radar_chart(data, features)
    create_boxplots(data, features)
    create_tsne_visualization(data, features)
    plot_risk_score_density(data)
    create_correlation_heatmap(data, features)
    
    print("All visualizations have been generated successfully!")

if __name__ == "__main__":
    main() 