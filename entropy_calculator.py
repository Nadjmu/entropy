import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binom

st.set_page_config(layout="wide", page_title="Distribution Entropy Visualizer")
sns.set_theme(style="whitegrid")

# Define distributions with their parameters
DISTRIBUTIONS = {
    "Bernoulli": {
        "parameters": {
            "p": {"min": 0.0, "max": 1.0, "default": 0.5, "step": 0.01, "label": "p (probability)"}
        }
    },
    "Binomial": {
        "parameters": {
            "n": {"min": 1, "max": 50, "default": 10, "step": 1, "label": "n (number of trials)"},
            "p": {"min": 0.0, "max": 1.0, "default": 0.5, "step": 0.01, "label": "p (probability)"}
        }
    }
}

# Calculate entropy for Bernoulli distribution
def calculate_bernoulli_entropy(p):
    if p == 0 or p == 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

# Calculate entropy for Binomial distribution
def calculate_binomial_entropy(n, p):
    probs = [binom.pmf(k, n, p) for k in range(n + 1)]
    entropy = 0
    for prob in probs:
        if prob > 0:
            entropy -= prob * np.log2(prob)
    return entropy

# Sidebar controls
st.sidebar.header("Choose the distribution")
distribution = st.sidebar.radio("", list(DISTRIBUTIONS.keys()))

# Main layout: two columns
left_col, mid, right_col = st.columns([0.45, 0.1, 0.45])

# Store parameter values
params = {}

with left_col:
    st.header(f"{distribution} distribution")
    
    if distribution == "Bernoulli":
        st.latex(r"\huge P(X=1) = p \quad , \quad P(X=0) = 1 - p")
    elif distribution == "Binomial":
        st.latex(r"\huge P(X=k) = \left(\!\begin{array}{c} n \\ k \end{array}\!\right) p^k (1-p)^{n-k} \quad , \quad k = 0, 1, \ldots, n")

    # Create sliders for all parameters
    for param_name, param_config in DISTRIBUTIONS[distribution]["parameters"].items():
        if param_config["step"] == 1:  # Integer parameter
            params[param_name] = st.slider(
                param_config["label"],
                param_config["min"],
                param_config["max"],
                param_config["default"],
                param_config["step"]
            )
        else:  # Float parameter
            params[param_name] = st.slider(
                param_config["label"],
                param_config["min"],
                param_config["max"],
                param_config["default"],
                param_config["step"]
            )
    
    # Plot the distribution
    fig, ax = plt.subplots(figsize=(6, 5))
    
    if distribution == "Bernoulli":
        p = params["p"]
        df_dist = pd.DataFrame({
            'Outcome': ['x=0', 'x=1'],
            'Probability': [1 - p, p]
        })
        
        sns.barplot(data=df_dist, x='Outcome', y='Probability', 
                    palette=['#6A8EAE', '#F3EAD3'], ax=ax, edgecolor='black', linewidth=2)
        
        ax.set_ylim(0, 1.0)
        
        for i, (idx, row) in enumerate(df_dist.iterrows()):
            ax.text(i, row['Probability'], f"{row['Probability']:.3f}",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    elif distribution == "Binomial":
        n = params["n"]
        p = params["p"]
        
        k_values = np.arange(0, n + 1)
        probabilities = [binom.pmf(k, n, p) for k in k_values]
        
        df_dist = pd.DataFrame({
            'k': [f'x={k}' for k in k_values],
            'Probability': probabilities
        })
        
        sns.barplot(data=df_dist, x='k', y='Probability',
                    palette=['#6A8EAE'] * len(k_values), ax=ax, edgecolor='black', linewidth=2)
        
        ax.set_ylim(0, max(probabilities) * 1.2 if max(probabilities) > 0 else 1.0)
        
        # Show only 3 labels if more than 10 values
        if n > 10:
            # Show first, middle, and last
            tick_positions = [0, n // 2, n]
            tick_labels = [f'x=0', f'x={n // 2}', f'x={n}']
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
    
    ax.set_ylabel('p(x)', fontsize=12)
    ax.set_xlabel(r'$x \in X$', fontsize=12)
    fig.tight_layout()
    st.pyplot(fig)

with right_col:
    st.header(f"Entropy")
    st.latex(r"\huge S = -\sum_{x \in X} p(x) \log p(x)")
    
    # Get number of parameters
    num_params = len(DISTRIBUTIONS[distribution]["parameters"])
    
    if num_params == 1:
        # Single plot for single parameter
        fig, ax = plt.subplots(figsize=(6, 5))
        
        param_name = list(DISTRIBUTIONS[distribution]["parameters"].keys())[0]
        param_config = DISTRIBUTIONS[distribution]["parameters"][param_name]
        
        if distribution == "Bernoulli":
            p_values = np.linspace(0.001, 0.999, 500)
            entropy_values = [calculate_bernoulli_entropy(p_val) for p_val in p_values]
            
            df_entropy = pd.DataFrame({param_name: p_values, 'Entropy': entropy_values})
            sns.lineplot(data=df_entropy, x=param_name, y='Entropy', ax=ax, linewidth=2.5, color='#3498db')
            
            current_entropy = calculate_bernoulli_entropy(params["p"])
            ax.plot(params["p"], current_entropy, 'ro', markersize=12, 
                   label=f'Current: S={current_entropy:.4f}', zorder=5)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.1)
        
        ax.set_xlabel(param_name, fontsize=12)
        ax.set_ylabel('S [bits]', fontsize=12)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        
    elif num_params == 2:
        # Two subplots for two parameters (vertically stacked)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
        
        param_names = list(DISTRIBUTIONS[distribution]["parameters"].keys())
        
        if distribution == "Binomial":
            n = params["n"]
            p = params["p"]
            
            # Plot 1: Entropy vs p (with fixed n)
            p_values = np.linspace(0.001, 0.999, 200)
            entropy_vs_p = [calculate_binomial_entropy(n, p_val) for p_val in p_values]
            
            df_entropy_p = pd.DataFrame({'p': p_values, 'Entropy': entropy_vs_p})
            sns.lineplot(data=df_entropy_p, x='p', y='Entropy', ax=ax1, linewidth=2.5, color='#3498db')
            
            current_entropy = calculate_binomial_entropy(n, p)
            ax1.plot(p, current_entropy, 'ro', markersize=12, 
                    label=f'Current: S={current_entropy:.4f}', zorder=5)
            
            ax1.set_xlabel('p', fontsize=12)
            ax1.set_ylabel('S [bits]', fontsize=12)
            ax1.set_title(f'S(p,n={n})', fontsize=12, fontweight='bold')
            ax1.set_xlim(0, 1)
            ax1.legend(fontsize=9, loc='upper right')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Entropy vs n (with fixed p)
            n_values = np.arange(1, 51)
            entropy_vs_n = [calculate_binomial_entropy(n_val, p) for n_val in n_values]
            
            df_entropy_n = pd.DataFrame({'n': n_values, 'Entropy': entropy_vs_n})
            sns.lineplot(data=df_entropy_n, x='n', y='Entropy', ax=ax2, linewidth=2.5, color='#e74c3c')
            
            ax2.plot(n, current_entropy, 'ro', markersize=12, 
                    label=f'Current: S={current_entropy:.4f}', zorder=5)
            
            ax2.set_xlabel('n', fontsize=12)
            ax2.set_ylabel('S [bits]', fontsize=12)
            ax2.set_title(f'S(n, p={p:.2f})', fontsize=12, fontweight='bold')
            ax2.legend(fontsize=9, loc='upper right')
            ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
        st.pyplot(fig)
    
    elif num_params > 2:
        # For 3+ parameters, create grid of plots
        num_cols = 2
        num_rows = (num_params + 1) // 2
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 5 * num_rows))
        axes = axes.flatten() if num_params > 2 else [axes]
        
        # This is a placeholder for future distributions with 3+ parameters
        for idx, (param_name, param_config) in enumerate(DISTRIBUTIONS[distribution]["parameters"].items()):
            ax = axes[idx]
            ax.text(0.5, 0.5, f'Entropy vs {param_name}', ha='center', va='center', fontsize=12)
        
        fig.tight_layout()
        st.pyplot(fig)