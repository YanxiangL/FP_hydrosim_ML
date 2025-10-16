import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

def clean_equation(equation_str):
    """Clean and simplify equations for display"""
    if pd.isna(equation_str):
        return "N/A"
    
    # Don't do heavy processing here, just clean the raw equation
    equation_str = str(equation_str).strip()
    
    # Clean LaTeX formatting but preserve the full equation
    eq = re.sub(r'\\left\(', '(', equation_str)
    eq = re.sub(r'\\right\)', ')', eq)
    eq = re.sub(r'\\log\{\\left\(', 'log(', eq)
    eq = re.sub(r'\\right\)\}', ')', eq)
    eq = re.sub(r'log_\{([^}]+)\}', r'log(\1)', eq)
    eq = re.sub(r'sin_\{([^}]+)\}', r'sin(\1)', eq)
    eq = re.sub(r'cos_\{([^}]+)\}', r'cos(\1)', eq)
    eq = re.sub(r'observed_\{([^}]+)\}', r'observed_\1', eq)
    eq = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', eq)
    eq = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', eq)
    eq = re.sub(r'\{([^}]+)\}', r'(\1)', eq)
    eq = re.sub(r'\\', '', eq)
    
    # Simplify variable names
    eq = eq.replace('observed velocity', 'v_obs')
    eq = eq.replace('observed_velocity', 'v_obs')  
    eq = eq.replace('velocity norm', 'v_norm')
    eq = eq.replace('observed_velocity norm', 'v_obs_norm')
    eq = eq.replace('inclination', 'inc')
    
    return eq

def truncate_equation(equation_str, max_length=80):
    """Truncate long equations for legend display"""
    if len(equation_str) <= max_length:
        return equation_str
    
    # Try to truncate at a sensible point (operators, parentheses, or spaces)
    for i in range(max_length, max_length//2, -1):
        if equation_str[i] in ['+', '-', '*', '/', ')', ' ', '(']:
            return equation_str[:i] + "..."
    
    return equation_str[:max_length-3] + "..."

# Read the CSV file
df = pd.read_csv('tf_results_math_csv.csv')

# Process each model separately and create individual plots
for model_idx, model_name in enumerate(df['model_name'].unique()):
    model_data = df[df['model_name'] == model_name].copy()
    
    # Create individual figure for this model
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig.suptitle(f'Tully-Fisher Symbolic Regression Results\n{model_name.replace("_", " ").title()} - Performance vs Complexity', 
                 fontsize=14, fontweight='bold')
    
    # Extract complexity, score, and equations from the CSV directly
    complexities = []
    scores = []
    equations = []
    
    # Process each row in the model data
    for _, row in model_data.iterrows():
        try:
            complexity_val = row.get('complexity')
            equation_val = row.get('equation', 'N/A')
            
            # Prioritize score metric over R² (score varies properly)
            if 'score' in row and not pd.isna(row['score']) and row['score'] is not None:
                metric_val = float(row['score'])
            elif 'r2_individual' in row and not pd.isna(row['r2_individual']):
                metric_val = float(row['r2_individual'])
            else:
                metric_val = float(row['r2_overall'])
            
            # Only include rows with valid complexity and metric values
            if not pd.isna(complexity_val) and complexity_val is not None and not pd.isna(metric_val):
                complexities.append(int(complexity_val))
                scores.append(metric_val)
                equations.append(clean_equation(str(equation_val)))
                
        except Exception as e:
            print(f"Warning: Error processing row for {model_name}: {e}")
            continue
    
    # If we have no valid data, use fallback
    if not complexities:
        print(f"No valid data found for {model_name}, using fallback")
        complexities = [5]  # Default complexity
        scores = [model_data['r2_overall'].iloc[0] if not model_data.empty else 0.5]
        equations = [f"{model_name} equation"]
    
    # Create unique colors for each bar using tab20 colormap for better distinction
    n_bars = len(complexities)
    if n_bars > 1:
        colors = plt.cm.tab20(np.linspace(0, 1, n_bars))
    else:
        colors = ['steelblue']
    
    # Create the bar plot with unique colors for each bar
    bars = ax.bar(complexities, scores, color=colors, 
                  alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add legend with colored patches and longer equations (only for displayed bars)
    legend_handles = []
    for i, (complexity, score, equation) in enumerate(zip(complexities, scores, equations)):
        color = bars[i].get_facecolor() if i < len(bars) else colors[0] 
        truncated_eq = truncate_equation(equation, 75)  # Increased from 35 to 75
        label = f"C{complexity} (Score: {score:.2e}): {truncated_eq}"
        patch = mpatches.Patch(color=color, label=label)
        legend_handles.append(patch)
    
    # Only show legend if we have meaningful equations to display
    if legend_handles and len(legend_handles) <= 15:  # Limit to avoid overcrowding
        ax.legend(handles=legend_handles, loc='upper right', fontsize=8, 
                  title="Equations by Complexity", title_fontsize=9, 
                  framealpha=0.9)
    
    # Customize the plot
    ax.set_xlabel('Model Complexity', fontsize=12)
    ax.set_ylabel('Performance Score', fontsize=12)
    ax.set_title(f'{model_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Set appropriate y-axis limits
    if scores:
        y_min = min(scores)
        y_max = max(scores)
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.3 * y_range)
    
    # Set x-axis to show integer ticks
    if complexities:
        x_min, x_max = min(complexities), max(complexities)
        if x_max > x_min:
            ax.set_xticks(range(x_min, x_max + 1))
        else:
            ax.set_xticks([x_min])
    
    # Save the plot
    plt.tight_layout()
    output_dir = "../plots"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/tully_fisher_{model_name.lower()}_results.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved plot: {filename}")
    plt.show()

# Process each model separately and create individual loss plots
for model_idx, model_name in enumerate(df['model_name'].unique()):
    model_data = df[df['model_name'] == model_name].copy()
    
    # Create individual figure for this model (Loss vs Complexity)
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig.suptitle(f'Tully-Fisher Symbolic Regression Results\n{model_name.replace("_", " ").title()} - Loss vs Complexity', 
                 fontsize=14, fontweight='bold')
    
    # Extract complexity, loss, and equations from the CSV directly
    complexities = []
    losses = []
    equations = []
    
    # Process each row in the model data
    for _, row in model_data.iterrows():
        try:
            complexity_val = row.get('complexity')
            equation_val = row.get('equation', 'N/A')
            
            # Look for loss metrics in the data
            if 'loss' in row and not pd.isna(row['loss']) and row['loss'] is not None:
                loss_val = float(row['loss'])
            elif 'mse_individual' in row and not pd.isna(row['mse_individual']):
                loss_val = float(row['mse_individual'])
            elif 'rmse_individual' in row and not pd.isna(row['rmse_individual']):
                loss_val = float(row['rmse_individual'])
            elif 'rmse_overall' in row and not pd.isna(row['rmse_overall']):
                loss_val = float(row['rmse_overall'])
            else:
                # Use 1 - R² as a proxy for loss if no explicit loss metric
                if 'score' in row and not pd.isna(row['score']) and row['score'] is not None:
                    loss_val = 1.0 - float(row['score'])
                elif 'r2_individual' in row and not pd.isna(row['r2_individual']):
                    loss_val = 1.0 - float(row['r2_individual'])
                else:
                    loss_val = 1.0 - float(row['r2_overall'])
            
            # Only include rows with valid complexity and loss values
            if not pd.isna(complexity_val) and complexity_val is not None and not pd.isna(loss_val):
                complexities.append(int(complexity_val))
                losses.append(loss_val)
                equations.append(clean_equation(str(equation_val)))
                
        except Exception as e:
            print(f"Warning: Error processing row for {model_name} (loss plot): {e}")
            continue
    
    # If we have no valid data, use fallback
    if not complexities:
        print(f"No valid loss data found for {model_name}, using fallback")
        complexities = [5]  # Default complexity
        losses = [0.5]  # Default loss
        equations = [f"{model_name} equation"]
    
    # Create unique colors for each bar using tab20 colormap for better distinction
    n_bars = len(complexities)
    if n_bars > 1:
        colors = plt.cm.tab20(np.linspace(0, 1, n_bars))
    else:
        colors = ['steelblue']
    
    # Create the bar plot with unique colors for each bar
    bars = ax.bar(complexities, losses, color=colors, 
                  alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add legend with colored patches and longer equations (only for displayed bars)
    legend_handles = []
    for i, (complexity, loss, equation) in enumerate(zip(complexities, losses, equations)):
        color = bars[i].get_facecolor() if i < len(bars) else colors[0] 
        truncated_eq = truncate_equation(equation, 75)  # Increased from 35 to 75
        label = f"C{complexity} (Loss: {loss:.2e}): {truncated_eq}"
        patch = mpatches.Patch(color=color, label=label)
        legend_handles.append(patch)
    
    # Only show legend if we have meaningful equations to display
    if legend_handles and len(legend_handles) <= 15:  # Limit to avoid overcrowding
        ax.legend(handles=legend_handles, loc='upper right', fontsize=8, 
                  title="Equations by Complexity", title_fontsize=9, 
                  framealpha=0.9)
    
    # Customize the plot
    ax.set_xlabel('Model Complexity', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'{model_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Set appropriate y-axis limits
    if losses:
        y_min = min(losses)
        y_max = max(losses)
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.3 * y_range)
    
    # Set x-axis to show integer ticks
    if complexities:
        x_min, x_max = min(complexities), max(complexities)
        if x_max > x_min:
            ax.set_xticks(range(x_min, x_max + 1))
        else:
            ax.set_xticks([x_min])
    
    # Save the plot
    plt.tight_layout()
    output_dir = "../plots"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/tully_fisher_{model_name.lower()}_loss_results.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved loss plot: {filename}")
    plt.show()

# Print summary statistics
print("="*60)
print("TULLY-FISHER SYMBOLIC REGRESSION SUMMARY")
print("="*60)

for model_name in df['model_name'].unique():
    model_data = df[df['model_name'] == model_name]
    if not model_data.empty:
        sample_row = model_data.iloc[0]
        print(f"\n{model_name.replace('_', ' ').title()}:")
        print(f"  Overall R² Score: {sample_row['r2_overall']:.4f}")
        print(f"  Overall RMSE: {sample_row['rmse_overall']:.4f}")
        print(f"  Number of equations: {len(model_data)}")
    
print("\n" + "="*60)