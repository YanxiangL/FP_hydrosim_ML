import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import itertools
import warnings
warnings.filterwarnings('ignore')

class TullyFisherSymbolicRegression:
    def __init__(self, random_state=42):
        """
        Initialize the symbolic regression system for Tully-Fisher relation
        
        Parameters:
        random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Define function primitives for symbolic regression
        self.primitives = {
            'add': lambda x, y: x + y,
            'sub': lambda x, y: x - y,
            'mul': lambda x, y: x * y,
            'div': lambda x, y: np.where(np.abs(y) > 1e-10, x / y, np.nan),
            'pow': lambda x, y: np.where((x > 0) & (np.abs(y) < 10), np.power(x, y), np.nan),
            'log': lambda x: np.where(x > 0, np.log10(x), np.nan),
            'exp': lambda x: np.where(np.abs(x) < 10, np.exp(x), np.nan),
            'sqrt': lambda x: np.where(x >= 0, np.sqrt(x), np.nan),
            'square': lambda x: x**2,
            'inv': lambda x: np.where(np.abs(x) > 1e-10, 1/x, np.nan)
        }
        
        # Store results
        self.results = []
        
    def load_data(self, filename='tully_fisher_simulated_dataset.csv'):
        """
        Load the simulated Tully-Fisher dataset
        
        Parameters:
        filename (str): Name of the dataset file (looks in ../input/ directory)
        
        Returns:
        pandas.DataFrame: Loaded dataset
        """
        # Construct path to input directory
        input_path = os.path.join('..', 'input', filename)
        
        try:
            df = pd.read_csv(input_path)
            print(f"✓ Loaded dataset with {len(df)} galaxies from {input_path}")
            print(f"✓ Available columns: {list(df.columns)}")
            
            # Validate essential columns
            required_columns = ['corrected_velocity', 'observed_velocity', 'stellar_mass', 
                              'distance', 'absolute_magnitude']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"❌ Missing required columns: {missing_columns}")
                return None
                
            print(f"✓ All required columns present")
            return df
            
        except FileNotFoundError:
            print(f"❌ Dataset file {input_path} not found.")
            print("Please run tf_simulation.py first to generate the dataset.")
            return None
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
            
    def prepare_features(self, df):
        """
        Prepare feature variables for symbolic regression
        
        Parameters:
        df (pandas.DataFrame): Input dataset
        
        Returns:
        dict: Dictionary of feature arrays
        """
        features = {}
        
        # Primary features
        features['v'] = df['corrected_velocity'].values  # Corrected velocity
        features['v_obs'] = df['observed_velocity'].values  # Observed velocity
        features['M_star'] = df['stellar_mass'].values  # Stellar mass
        features['D'] = df['distance'].values  # Distance
        features['M_abs'] = df['absolute_magnitude'].values  # Target variable
        
        # Derived features
        features['log_v'] = np.log10(features['v'])
        features['log_v_obs'] = np.log10(features['v_obs'])
        features['log_M_star'] = np.log10(features['M_star'])
        features['log_D'] = np.log10(features['D'])
        
        # Normalized features
        features['v_norm'] = features['v'] / 200.0  # Normalize to 200 km/s
        features['v_obs_norm'] = features['v_obs'] / 200.0
        features['M_star_norm'] = features['M_star'] / 1e10  # Normalize to 10^10 M_sun
        features['D_norm'] = features['D'] / 30.0  # Normalize to 30 Mpc
        
        return features
    
    def evaluate_expression(self, expression, features, params=None):
        """
        Evaluate a symbolic expression with given features
        
        Parameters:
        expression (str): Mathematical expression to evaluate
        features (dict): Feature dictionary
        params (dict): Parameter dictionary for fitting
        
        Returns:
        numpy.array: Evaluated expression values
        """
        try:
            # Start with the original expression
            eval_expr = expression
            
            # Replace parameters first (with word boundaries to avoid substring issues)
            if params:
                for param, value in params.items():
                    # Use word boundary regex to replace only whole parameter names
                    pattern = r'\b' + re.escape(param) + r'\b'
                    eval_expr = re.sub(pattern, str(value), eval_expr)
            
            # Replace variables using regex (sort by length desc to handle longer names first)
            sorted_vars = sorted(features.keys(), key=len, reverse=True)
            for var in sorted_vars:
                # Use word boundary regex for variables too
                pattern = r'\b' + re.escape(var) + r'\b'
                eval_expr = re.sub(pattern, f"features['{var}']", eval_expr)
            
            # Evaluate expression safely
            result = eval(eval_expr, {"__builtins__": {}}, 
                         {"features": features, "np": np, "log10": np.log10})
            
            # Handle NaN values
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                return np.full_like(result, np.nan)
            
            return result
            
        except Exception as e:
            return np.full(len(list(features.values())[0]), np.nan)
    
    def fit_parameters(self, expression, features, target, param_names):
        """
        Fit parameters for a given expression using least squares
        
        Parameters:
        expression (str): Expression with parameters to fit
        features (dict): Feature dictionary
        target (array): Target values
        param_names (list): Names of parameters to fit
        
        Returns:
        dict: Fitted parameters
        """
        def objective(params):
            param_dict = dict(zip(param_names, params))
            predicted = self.evaluate_expression(expression, features, param_dict)
            
            # Handle NaN values
            mask = ~np.isnan(predicted) & ~np.isnan(target)
            if np.sum(mask) < 10:  # Need minimum data points
                return 1e10
            
            return np.sum((predicted[mask] - target[mask])**2)
        
        # Initial parameter guesses
        initial_params = [1.0] * len(param_names)
        
        # Optimize parameters
        try:
            result = minimize(objective, initial_params, method='L-BFGS-B',
                            bounds=[(-10, 10)] * len(param_names))
            if result.success:
                return dict(zip(param_names, result.x))
            else:
                return dict(zip(param_names, initial_params))
        except:
            return dict(zip(param_names, initial_params))
    
    def calculate_complexity(self, expression):
        """
        Calculate complexity score for an expression
        
        Parameters:
        expression (str): Mathematical expression
        
        Returns:
        int: Complexity score
        """
        # Count operators, variables, and constants
        operators = ['+', '-', '*', '/', '**', 'log10', 'exp', 'sqrt']
        complexity = 0
        
        # Count operators
        for op in operators:
            complexity += expression.count(op)
        
        # Count variables (approximate)
        variables = ['v', 'M_star', 'D', 'log_v', 'log_M_star', 'log_D']
        for var in variables:
            complexity += expression.count(var)
        
        # Count parameters
        params = ['a', 'b', 'c', 'd', 'e']
        for param in params:
            complexity += expression.count(param)
        
        return complexity
    
    def generate_candidate_expressions(self):
        """
        Generate candidate expressions for Tully-Fisher relation
        
        Returns:
        list: List of (expression, param_names, description) tuples
        """
        candidates = []
        
        # 1. Linear fits
        candidates.append(("a + b * log_v", ["a", "b"], "Linear TF (log v)"))
        candidates.append(("a + b * log_v_obs", ["a", "b"], "Linear TF (log v_obs)"))
        candidates.append(("a + b * v_norm", ["a", "b"], "Linear TF (v normalized)"))
        
        # 2. Classic Tully-Fisher variations
        candidates.append(("a - b * log_v", ["a", "b"], "Classic TF"))
        candidates.append(("a - b * log_v_obs", ["a", "b"], "Classic TF (observed)"))
        candidates.append(("a - b * v_norm", ["a", "b"], "Classic TF (normalized)"))
        
        # 3. Power law variations (using log space)
        candidates.append(("a + b * log_v * log_v", ["a", "b"], "Power law TF"))
        candidates.append(("a * v_norm", ["a"], "Linear power TF (normalized)"))
        candidates.append(("a + b * v_norm + c * v_norm * v_norm", ["a", "b", "c"], "Quadratic normalized TF"))
        
        # 4. Multi-parameter relations
        candidates.append(("a - b * log_v + c * log_M_star", ["a", "b", "c"], "TF + stellar mass"))
        candidates.append(("a - b * log_v + c * log_D", ["a", "b", "c"], "TF + distance"))
        candidates.append(("a - b * log_v + c * log_v * log_v", ["a", "b", "c"], "TF + velocity squared"))
        
        # 5. More complex relations
        candidates.append(("a - b * log_v + c * log_M_star + d * log_D", 
                         ["a", "b", "c", "d"], "TF + mass + distance"))
        candidates.append(("a - b * log_v + c * log_M_star + d * log_v * log_v", 
                         ["a", "b", "c", "d"], "TF + mass + velocity²"))
        candidates.append(("a - b * log_v + c * log_M_star + d * log_D + e * log_v * log_v", 
                         ["a", "b", "c", "d", "e"], "Full systematic model"))
        
        # 6. Simpler non-linear terms
        candidates.append(("a - b * log_v + c * log_v * log_v", 
                         ["a", "b", "c"], "TF + quadratic velocity"))
        candidates.append(("a - b * log_v + c * log_v * log_M_star", 
                         ["a", "b", "c"], "TF + velocity-mass interaction"))
        
        # 7. Normalized versions
        candidates.append(("a - b * v_norm + c * M_star_norm", ["a", "b", "c"], "Normalized TF + mass"))
        candidates.append(("a - b * v_norm + c * D_norm", ["a", "b", "c"], "Normalized TF + distance"))
        
        return candidates
    
    def evaluate_model(self, expression, features, target, param_names, description):
        """
        Evaluate a single model expression
        
        Parameters:
        expression (str): Expression to evaluate
        features (dict): Feature dictionary
        target (array): Target values
        param_names (list): Parameter names
        description (str): Model description
        
        Returns:
        dict: Evaluation results
        """
        # Split data into train/test
        n_total = len(target)
        train_size = int(0.8 * n_total)
        indices = np.random.permutation(n_total)
        train_idx, test_idx = indices[:train_size], indices[train_size:]
        
        # Create train/test features
        train_features = {k: v[train_idx] for k, v in features.items()}
        test_features = {k: v[test_idx] for k, v in features.items()}
        train_target = target[train_idx]
        test_target = target[test_idx]
        
        # Fit parameters
        fitted_params = self.fit_parameters(expression, train_features, 
                                          train_target, param_names)
        
        # Evaluate on test set
        test_pred = self.evaluate_expression(expression, test_features, fitted_params)
        
        # Calculate metrics
        mask = ~np.isnan(test_pred) & ~np.isnan(test_target)
        if np.sum(mask) < 5:
            return None
        
        mse = mean_squared_error(test_target[mask], test_pred[mask])
        r2 = r2_score(test_target[mask], test_pred[mask])
        rmse = np.sqrt(mse)
        
        # Calculate complexity
        complexity = self.calculate_complexity(expression)
        
        # Performance score (higher is better)
        performance = r2 - 0.01 * complexity  # Penalize complexity
        
        return {
            'expression': expression,
            'description': description,
            'parameters': fitted_params,
            'complexity': complexity,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'performance': performance,
            'n_test': np.sum(mask)
        }
    
    def run_symbolic_regression(self, df):
        """
        Run symbolic regression on the dataset
        
        Parameters:
        df (pandas.DataFrame): Input dataset
        
        Returns:
        list: List of evaluation results
        """
        print("Preparing features...")
        features = self.prepare_features(df)
        target = features['M_abs']
        
        print("Generating candidate expressions...")
        candidates = self.generate_candidate_expressions()
        
        print(f"Evaluating {len(candidates)} candidate expressions...")
        results = []
        
        for i, (expression, param_names, description) in enumerate(candidates):
            print(f"  {i+1}/{len(candidates)}: {description}")
            
            result = self.evaluate_model(expression, features, target, 
                                       param_names, description)
            
            if result is not None:
                results.append(result)
        
        # Sort by performance
        results.sort(key=lambda x: x['performance'], reverse=True)
        
        self.results = results
        return results
    
    def plot_results(self, results, figsize=(12, 8)):
        """
        Create complexity vs performance plot
        
        Parameters:
        results (list): Evaluation results
        figsize (tuple): Figure size
        """
        if not results:
            print("No results to plot")
            return
        
        # Extract data for plotting
        complexities = [r['complexity'] for r in results]
        performances = [r['performance'] for r in results]
        r2_scores = [r['r2'] for r in results]
        descriptions = [r['description'] for r in results]
        
        # Create main plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Complexity vs Performance
        scatter = ax1.scatter(complexities, performances, 
                            c=r2_scores, cmap='viridis', 
                            s=100, alpha=0.7, edgecolors='black')
        
        ax1.set_xlabel('Model Complexity')
        ax1.set_ylabel('Performance Score (R² - 0.01×Complexity)')
        ax1.set_title('Symbolic Regression Results:\nComplexity vs Performance')
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('R² Score')
        
        # Annotate best models
        for i, (comp, perf, desc) in enumerate(zip(complexities[:5], 
                                                  performances[:5], 
                                                  descriptions[:5])):
            ax1.annotate(f"{i+1}", (comp, perf), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, fontweight='bold')
        
        # Plot 2: R² vs Complexity
        ax2.scatter(complexities, r2_scores, 
                   c=performances, cmap='plasma', 
                   s=100, alpha=0.7, edgecolors='black')
        
        ax2.set_xlabel('Model Complexity')
        ax2.set_ylabel('R² Score')
        ax2.set_title('Model Accuracy vs Complexity')
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar2 = plt.colorbar(ax2.collections[0], ax=ax2)
        cbar2.set_label('Performance Score')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def print_results_table(self, results, top_n=10):
        """
        Print a formatted table of results
        
        Parameters:
        results (list): Evaluation results
        top_n (int): Number of top results to show
        """
        if not results:
            print("No results to display")
            return
        
        print(f"\n{'='*80}")
        print(f"TOP {top_n} SYMBOLIC REGRESSION RESULTS")
        print(f"{'='*80}")
        
        print(f"{'Rank':<4} {'Description':<25} {'R²':<6} {'RMSE':<6} {'Complexity':<10} {'Performance':<11}")
        print("-" * 80)
        
        for i, result in enumerate(results[:top_n]):
            print(f"{i+1:<4} {result['description']:<25} "
                  f"{result['r2']:<6.3f} {result['rmse']:<6.3f} "
                  f"{result['complexity']:<10} {result['performance']:<11.3f}")
        
        print("\nTop 3 expressions with fitted parameters:")
        print("-" * 50)
        
        for i, result in enumerate(results[:3]):
            print(f"\n{i+1}. {result['description']}")
            print(f"   Expression: {result['expression']}")
            print(f"   Parameters: {result['parameters']}")
            print(f"   R² = {result['r2']:.4f}, RMSE = {result['rmse']:.4f}")

def main():
    """
    Main function to run the symbolic regression analysis
    """
    print("=== Tully-Fisher Symbolic Regression Analysis ===")
    print("Attempting to rediscover the Tully-Fisher relation using AI...")
    
    # Initialize the symbolic regression system
    sr = TullyFisherSymbolicRegression(random_state=42)
    
    # Load the dataset
    df = sr.load_data('tully_fisher_simulated_dataset.csv')
    
    if df is None:
        print("\n❌ Failed to load dataset. Cannot proceed.")
        print("Please run tf_simulation.py first to generate the dataset.")
        return
    
    # Validate dataset size
    if len(df) < 100:
        print(f"\n❌ Dataset too small ({len(df)} galaxies). Need at least 100 galaxies.")
        return
    
    print(f"\n✓ Dataset validation passed. Proceeding with {len(df)} galaxies...")
    
    # Run symbolic regression
    print("\n🔍 Running symbolic regression to discover distance relations...")
    results = sr.run_symbolic_regression(df)
    
    if results:
        print(f"\n✅ SUCCESS! Discovered {len(results)} viable expressions")
        
        # Print results table
        sr.print_results_table(results)
        
        # Create plots
        fig = sr.plot_results(results)
        
        # Save results to models directory
        models_dir = os.path.join('..', 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        results_path = os.path.join(models_dir, 'symbolic_regression_results.csv')
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_path, index=False)
        print(f"\n💾 Results saved to '{results_path}'")
        
        # Show specific insights
        print(f"\n📊 Analysis Summary:")
        print(f"   • Evaluated {len(results)} successful models")
        print(f"   • Best R² score: {results[0]['r2']:.4f}")
        print(f"   • Best performance score: {results[0]['performance']:.4f}")
        print(f"   • Complexity range: {min(r['complexity'] for r in results)} to {max(r['complexity'] for r in results)}")
        
        # Check if we rediscovered the TF relation
        print(f"\n🎯 Tully-Fisher Relation Discovery Assessment:")
        best_model = results[0]
        if 'log_v' in best_model['expression'] and best_model['r2'] > 0.5:
            print(f"   ✅ Successfully rediscovered a logarithmic velocity-magnitude relation!")
            print(f"   ✅ Best expression: {best_model['expression']}")
            print(f"   ✅ This resembles the classical Tully-Fisher relation!")
        else:
            print(f"   ⚠️  Top model may not be the classical TF relation")
            print(f"   ⚠️  Consider expanding the search space or adjusting parameters")
        
    else:
        print("\n❌ No successful models found.")
        print("This might be due to:")
        print("   • Insufficient data quality")
        print("   • Too restrictive search parameters")
        print("   • Numerical issues in evaluation")
        print("   • Need to expand the expression search space")

if __name__ == "__main__":
    main()
