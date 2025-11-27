import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

st.set_page_config(layout="wide", page_title="Distribution Entropy Visualizer")
sns.set_theme(style="whitegrid")

# Sidebar controls
st.sidebar.header("Choose the distribution")
distribution = st.sidebar.selectbox("Distribution", ["Bernoulli"], index=0)

# Calculate entropy for Bernoulli distribution
# S = -p*log(p) - (1-p)*log(1-p)
def calculate_bernoulli_entropy(p):
    if p == 0 or p == 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


# Main layout: two columns
left_col, right_col = st.columns(2)

with left_col:
    st.header("Bernoulli distribution")
    
    # Bernoulli parameter slider inside left window
    p = st.slider("p (probability)", 0.0, 1.0, 0.5, 0.01)
    
    # Create Bernoulli distribution plot with seaborn
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Prepare data for seaborn
    df_bernoulli = pd.DataFrame({
        'Outcome': ['Failure (0)', 'Success (1)'],
        'Probability': [1 - p, p]
    })
    
    # Create barplot with seaborn
    sns.barplot(data=df_bernoulli, x='Outcome', y='Probability', 
                palette = ['#6A8EAE', '#F3EAD3'], ax=ax, edgecolor='black', linewidth=2)
    
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_xlabel('Outcome', fontsize=12)
    ax.set_ylim(0, 1.0)
    
    # Add probability values on top of bars
    for i, (idx, row) in enumerate(df_bernoulli.iterrows()):
        ax.text(i, row['Probability'], f"{row['Probability']:.3f}",
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    fig.tight_layout()
    st.pyplot(fig)

with right_col:
    st.latex(r"\huge S = -\sum_{x \in X} p(x) \log p(x)")
    
    # Create entropy plot
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Generate p values and corresponding entropy
    p_values = np.linspace(0.001, 0.999, 500)
    entropy_values = [-p_val * np.log2(p_val) - (1 - p_val) * np.log2(1 - p_val) 
                      for p_val in p_values]
    
    # Create dataframe for seaborn
    df_entropy = pd.DataFrame({
        'p': p_values,
        'Entropy': entropy_values
    })
    
    # Plot entropy curve
    sns.lineplot(data=df_entropy, x='p', y='Entropy', ax=ax, linewidth=2.5, color='#3498db')
    
    # Mark current entropy value
    current_entropy = calculate_bernoulli_entropy(p)
    ax.plot(p, current_entropy, 'ro', markersize=12, label=f'Current: S={current_entropy:.4f}', zorder=5)
    
    ax.set_xlabel('p', fontsize=12)
    ax.set_ylabel('S [bits]', fontsize=12)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add formula as text
    #ax.text(0.5, 0.05, r'$S = -p \log_2(p) - (1-p) \log_2(1-p),ha='center', fontsize=11, transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    fig.tight_layout()
    st.pyplot(fig)