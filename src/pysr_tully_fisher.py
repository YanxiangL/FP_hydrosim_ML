#!/usr/bin/env python3
"""
Comprehensive PySR Symbolic Regression System for Tully-Fisher Relation Discovery

This system uses PySR to automatically discover the Tully-Fisher relation and
hidden systematic effects from simulated galaxy data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings
import os
import subprocess
warnings.filterwarnings('ignore')

# Import PySR
try:
    from pysr import PySRRegressor
    PYSR_AVAILABLE = True
except ImportError:
    print("PySR not available. Install with: pip install pysr")
    PYSR_AVAILABLE = False

class TullyFisherSymbolicRegression:
    """
    Comprehensive symbolic regression system for discovering the Tully-Fisher relation
    """
    
    def __init__(self, data_file=None, random_state=42):
        self.data_file = data_file
        self.random_state = random_state
        self.data = None
        self.features = None
        self.target = None
        self.models = {}
        self.results = {}
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_data(self, data_file=None):
        """Load and preprocess the Tully-Fisher dataset"""
        if data_file:
            self.data_file = data_file
            
        if not self.data_file:
            print("No data file specified. Please provide a CSV file.")
            return
            
        try:
            self.data = pd.read_csv(self.data_file)
            print(f"Loaded data with shape: {self.data.shape}")
            print(f"Columns: {list(self.data.columns)}")
            
            # Display basic statistics
            print("\nBasic Statistics:")
            print(self.data.describe())
            
            # Check for missing values
            missing = self.data.isnull().sum()
            if missing.any():
                print(f"\nMissing values:\n{missing[missing > 0]}")
                
        except Exception as e:
            print(f"Error loading data: {e}")
            return
            
    def prepare_features(self):
        """Prepare features for symbolic regression"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
            
        # Create feature matrix with both observed and true values
        feature_columns = [
            'Velocity_width', 'observed_velocity', 'corrected_velocity',
            'stellar_mass', 'halo_mass', 'distance', 'inclination',
            'surface_brightness', 'luminosity'
        ]
        
        # Filter available columns
        available_cols = [col for col in feature_columns if col in self.data.columns]
        
        # Create derived features
        derived_features = pd.DataFrame()
        
        # Log transforms (common in astronomy)
        for col in ['Velocity_width', 'observed_velocity', 'stellar_mass', 'halo_mass', 'distance']:
            if col in self.data.columns:
                derived_features[f'log_{col}'] = np.log10(self.data[col] + 1e-10)
                
        # Normalized features
        for col in ['Velocity_width', 'observed_velocity']:
            if col in self.data.columns:
                derived_features[f'{col}_norm'] = self.data[col] / 200.0  # Normalize by typical velocity
                
        # Include inclination corrections
        if 'inclination' in self.data.columns:
            derived_features['sin_inclination'] = np.sin(np.radians(self.data['inclination']))
            derived_features['cos_inclination'] = np.cos(np.radians(self.data['inclination']))
            
        # Surface brightness features
        if 'surface_brightness' in self.data.columns:
            derived_features['surface_brightness'] = self.data['surface_brightness']
            
        # Distance features
        if 'distance' in self.data.columns:
            derived_features['distance'] = self.data['distance']
            derived_features['distance_modulus'] = 5 * np.log10(self.data['distance']) - 5
            
        # Stellar and halo mass features
        for col in ['stellar_mass', 'halo_mass']:
            if col in self.data.columns:
                derived_features[col] = self.data[col]
                
        # Combine original and derived features
        self.features = pd.concat([
            self.data[available_cols],
            derived_features
        ], axis=1)
        
        # Set target variable (absolute magnitude - the classic TF target)
        if 'true_absolute_magnitude' in self.data.columns:
            self.target = self.data['true_absolute_magnitude']
        elif 'absolute_magnitude' in self.data.columns:
            self.target = self.data['absolute_magnitude']
        else:
            print("No suitable target variable found")
            return
            
        print(f"Prepared {self.features.shape[1]} features for {self.features.shape[0]} galaxies")
        print(f"Feature columns: {list(self.features.columns)}")
        
    def run_symbolic_regression(self, 
                               target_col=None,
                               feature_subset=None,
                               model_name="default",
                               **pysr_kwargs):
        """
        Run PySR symbolic regression
        """
        if not PYSR_AVAILABLE:
            print("PySR not available. Please install PySR.")
            return
            
        if self.features is None:
            print("Features not prepared. Please run prepare_features() first.")
            return
            
        # Select target
        if target_col:
            if target_col in self.data.columns:
                target = self.data[target_col]
            else:
                print(f"Target column '{target_col}' not found")
                return
        else:
            target = self.target
            
        # Select feature subset if specified
        if feature_subset:
            features = self.features[feature_subset]
        else:
            features = self.features
            
        # Remove any NaN values
        mask = ~(np.isnan(features).any(axis=1) | np.isnan(target))
        features = features[mask]
        target = target[mask]
        
        print(f"Running symbolic regression with {features.shape[1]} features and {features.shape[0]} samples")
        
        # Default PySR parameters optimized for Tully-Fisher discovery
        # Remove 'constraints' and 'complexity_of_operators' for PySR compatibility
        default_params = {
            'niterations': 100,
            'populations': 30,
            'population_size': 50,
            'binary_operators': ['+', '-', '*', '/', '^'],
            'unary_operators': ['log10', 'log', 'exp', 'sqrt', 'abs'],
            'maxsize': 25,
            'maxdepth': 8,
            'parsimony': 0.01,
            'adaptive_parsimony_scaling': 20,
            'tournament_selection_p': 0.86,
            'random_state': self.random_state
        }
        
        # Update with user parameters
        params = {**default_params, **pysr_kwargs}
        
        # Initialize PySR model
        model = PySRRegressor(**params)
        
        # Fit the model
        print("Fitting PySR model... This may take several minutes.")
        try:
            model.fit(features, target)
            
            # Store results
            self.models[model_name] = model
            
            # Calculate performance metrics
            predictions = model.predict(features)
            r2 = r2_score(target, predictions)
            rmse = np.sqrt(mean_squared_error(target, predictions))
            
            self.results[model_name] = {
                'model': model,
                'r2': r2,
                'rmse': rmse,
                'features': list(features.columns),
                'target': target_col if target_col else 'target',
                'n_samples': len(target)
            }
            
            print(f"Model '{model_name}' completed:")
            print(f"  R²: {r2:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  Best equation: {model.latex()}")
            
        except Exception as e:
            print(f"Error fitting model: {e}")
            
    def run_comprehensive_search(self):
        """
        Run multiple symbolic regression experiments to discover different aspects
        of the Tully-Fisher relation
        """
        print("Starting comprehensive symbolic regression search...")
        
        # 1. Classic Tully-Fisher (velocity vs absolute magnitude)
        velocity_features = ['log_Velocity_width', 'log_observed_velocity', 
                           'Velocity_width_norm', 'observed_velocity_norm']
        available_velocity = [f for f in velocity_features if f in self.features.columns]
        
        if available_velocity:
            print("\n1. Classic Tully-Fisher Relation")
            self.run_symbolic_regression(
                feature_subset=available_velocity,
                model_name="classic_TF",
                niterations=80,
                maxsize=15
            )
        
        # 2. Extended Tully-Fisher (including stellar mass)
        extended_features = available_velocity + ['log_stellar_mass', 'stellar_mass']
        available_extended = [f for f in extended_features if f in self.features.columns]
        
        if available_extended:
            print("\n2. Extended Tully-Fisher (with stellar mass)")
            self.run_symbolic_regression(
                feature_subset=available_extended,
                model_name="extended_TF",
                niterations=100,
                maxsize=20
            )
            
        # 3. Full systematic search (all available features)
        print("\n3. Full Systematic Search")
        self.run_symbolic_regression(
            model_name="full_systematic",
            niterations=120,
            maxsize=25
        )
        
        # 4. Distance-corrected Tully-Fisher
        distance_features = available_velocity + ['distance', 'distance_modulus', 'log_distance']
        available_distance = [f for f in distance_features if f in self.features.columns]
        
        if available_distance:
            print("\n4. Distance-corrected Tully-Fisher")
            self.run_symbolic_regression(
                feature_subset=available_distance,
                model_name="distance_corrected_TF",
                niterations=100,
                maxsize=20
            )
            
        # 5. Inclination-corrected search
        inclination_features = available_velocity + ['sin_inclination', 'cos_inclination', 'inclination']
        available_inclination = [f for f in inclination_features if f in self.features.columns]
        
        if available_inclination:
            print("\n5. Inclination-corrected Tully-Fisher")
            self.run_symbolic_regression(
                feature_subset=available_inclination,
                model_name="inclination_corrected_TF",
                niterations=100,
                maxsize=20
            )
            
    def analyze_results(self):
        """Analyze and compare results from all models"""
        if not self.results:
            print("No results to analyze. Please run symbolic regression first.")
            return
            
        print("\n" + "="*80)
        print("SYMBOLIC REGRESSION RESULTS ANALYSIS")
        print("="*80)
        
        # Create results summary
        summary_data = []
        for name, result in self.results.items():
            model = result['model']
            
            # Get the best equation
            latex_table = model.latex_table() if callable(getattr(model, 'latex_table', None)) else model.latex_table
            if hasattr(latex_table, 'iloc'):
                best_eq = latex_table.iloc[0]
            else:
                # Fallback: PySR <0.13 returns a string, not a DataFrame
                best_eq = {'complexity': None, 'equation': str(latex_table), 'latex_format': str(latex_table)}
            
            summary_data.append({
                'Model': name,
                'R²': result['r2'],
                'RMSE': result['rmse'],
                'Complexity': best_eq['complexity'],
                'Equation': best_eq['equation'],
                'LaTeX': best_eq['latex_format']
            })
            
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('R²', ascending=False)
        
        print("\nModel Performance Summary:")
        print(summary_df[['Model', 'R²', 'RMSE', 'Complexity']].to_string(index=False))
        
        print("\n" + "="*80)
        print("DISCOVERED EQUATIONS")
        print("="*80)
        
        for _, row in summary_df.iterrows():
            print(f"\n{row['Model'].upper()}:")
            print(f"  R² = {row['R²']:.4f}, RMSE = {row['RMSE']:.4f}")
            print(f"  Complexity: {row['Complexity']}")
            print(f"  Equation: {row['Equation']}")
            print(f"  LaTeX: {row['LaTeX']}")
            
    def visualize_results(self):
        """Create comprehensive visualizations of the results"""
        if not self.results:
            print("No results to visualize.")
            return
            
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Performance vs Complexity plot
        ax1 = plt.subplot(2, 3, 1)
        complexities = []
        r2_scores = []
        model_names = []
        
        for name, result in self.results.items():
            model = result['model']
            try:
                # Try to get complexity from the equations table
                equations_table = model.equations_
                if hasattr(equations_table, 'complexity'):
                    # Get the best equation (lowest loss or highest score)
                    best_idx = equations_table['loss'].idxmin() if 'loss' in equations_table.columns else 0
                    complexity = equations_table.loc[best_idx, 'complexity']
                else:
                    # Fallback: estimate complexity from equation string
                    best_eq = str(model.sympy())
                    # Count operators and variables as a complexity estimate
                    complexity = len([c for c in best_eq if c in '+-*/^()']) + len(set([c for c in best_eq if c.isalpha()]))
                    
                complexities.append(complexity)
                r2_scores.append(result['r2'])
                model_names.append(name)
            except Exception as e:
                print(f"Warning: Could not extract complexity for {name}: {e}")
                # Use a default complexity based on model name if extraction fails
                default_complexity = {'classic_TF': 5, 'extended_TF': 8, 'full_systematic': 15, 
                                    'distance_corrected_TF': 10, 'inclination_corrected_TF': 12}.get(name, 10)
                complexities.append(default_complexity)
                r2_scores.append(result['r2'])
                model_names.append(name)
        
        # Plot complexity vs performance
        if complexities and r2_scores:
            colors = plt.cm.tab10(np.linspace(0, 1, len(complexities)))
            scatter = ax1.scatter(complexities, r2_scores, s=100, alpha=0.7, c=colors)
            for i, name in enumerate(model_names):
                ax1.annotate(name.replace('_', ' '), (complexities[i], r2_scores[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # Set reasonable axis limits
            x_margin = (max(complexities) - min(complexities)) * 0.1 if len(set(complexities)) > 1 else 1
            y_margin = (max(r2_scores) - min(r2_scores)) * 0.1 if len(set(r2_scores)) > 1 else 0.1
            ax1.set_xlim(min(complexities) - x_margin, max(complexities) + x_margin)
            ax1.set_ylim(min(r2_scores) - y_margin, max(r2_scores) + y_margin)
        else:
            ax1.text(0.5, 0.5, 'No complexity data available', ha='center', va='center', fontsize=12)
            ax1.set_xlim(0, 20)
            ax1.set_ylim(0, 1)
            
        ax1.set_xlabel('Model Complexity')
        ax1.set_ylabel('R² Score')
        ax1.set_title('Performance vs Complexity')
        ax1.grid(True, alpha=0.3)
        
        # 2. Pareto frontier plot (Error vs Complexity)
        ax2 = plt.subplot(2, 3, 2)
        rmse_scores = [result['rmse'] for result in self.results.values()]
        
        if complexities and rmse_scores:
            colors = plt.cm.tab10(np.linspace(0, 1, len(complexities)))
            scatter2 = ax2.scatter(complexities, rmse_scores, s=100, alpha=0.7, c=colors)
            for i, name in enumerate(model_names):
                ax2.annotate(name.replace('_', ' '), (complexities[i], rmse_scores[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # Set reasonable axis limits
            x_margin = (max(complexities) - min(complexities)) * 0.1 if len(set(complexities)) > 1 else 1
            y_margin = (max(rmse_scores) - min(rmse_scores)) * 0.1 if len(set(rmse_scores)) > 1 else 0.1
            ax2.set_xlim(min(complexities) - x_margin, max(complexities) + x_margin)
            ax2.set_ylim(min(rmse_scores) - y_margin, max(rmse_scores) + y_margin)
        else:
            ax2.text(0.5, 0.5, 'No complexity data available', ha='center', va='center', fontsize=12)
            ax2.set_xlim(0, 20)
            ax2.set_ylim(0, 1)
            
        ax2.set_xlabel('Model Complexity')
        ax2.set_ylabel('RMSE')
        ax2.set_title('Error vs Complexity')
        ax2.grid(True, alpha=0.3)
        
        # 3. Best model prediction vs actual (only best model in main grid)
        if self.results:
            # Find best model by R^2
            sorted_models = sorted(self.results.items(), key=lambda x: x[1]['r2'], reverse=True)
            best_model_name, best_result = sorted_models[0]
            best_model = best_result['model']
            feature_names = getattr(best_model, 'feature_names_in_', None)
            if feature_names is not None:
                features_for_pred = self.features.loc[:, feature_names]
            else:
                features_for_pred = self.features
            ax3 = plt.subplot(2, 3, 3)
            best_predictions = best_model.predict(features_for_pred)
            min_val = min(self.target.min(), best_predictions.min())
            max_val = max(self.target.max(), best_predictions.max())
            ax3.scatter(self.target, best_predictions, alpha=0.6, s=20)
            ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            ax3.set_xlabel('True Absolute Magnitude')
            ax3.set_ylabel('Predicted Absolute Magnitude')
            ax3.set_title(f'Model: {best_model_name}')
            ax3.grid(True, alpha=0.3)
            
        # 4. Residuals plot (use best_predictions only)
        if self.results:
            ax4 = plt.subplot(2, 3, 4)
            residuals = self.target - best_predictions
            ax4.scatter(best_predictions, residuals, alpha=0.6, s=20)
            ax4.axhline(y=0, color='r', linestyle='--', alpha=0.8)
            ax4.set_xlabel('Predicted Absolute Magnitude')
            ax4.set_ylabel('Residuals')
            ax4.set_title('Residual Plot')
            ax4.grid(True, alpha=0.3)
            
        # 5. Feature importance (complexity contribution)
        # Use the best model (highest R²)
        if self.results:
            sorted_models = sorted(self.results.items(), key=lambda x: x[1]['r2'], reverse=True)
            best_model = sorted_models[0][1]['model']
            ax5 = plt.subplot(2, 3, 5)
            try:
                equation_str = str(best_model.sympy())
                # Count feature usage
                feature_counts = {}
                for col in self.features.columns:
                    if col in equation_str:
                        feature_counts[col] = equation_str.count(col)
                if feature_counts:
                    features_used = list(feature_counts.keys())
                    counts = list(feature_counts.values())
                    ax5.barh(features_used, counts)
                    ax5.set_xlabel('Usage Count in Best Equation')
                    ax5.set_title('Feature Usage in Best Model')
                else:
                    ax5.text(0.5, 0.5, 'No feature usage data', ha='center', va='center', fontsize=12)
            except Exception as e:
                ax5.text(0.5, 0.5, f'Error extracting features', ha='center', va='center', fontsize=10)
            ax5.set_title('Feature Usage Analysis')
                
        # 6. Model comparison radar chart
        ax6 = plt.subplot(2, 3, 6)
        
        # Create a simple bar chart of R² scores
        model_names_short = [name[:15] for name in model_names]  # Truncate names
        bars = ax6.bar(model_names_short, r2_scores, alpha=0.7)
        ax6.set_ylabel('R² Score')
        ax6.set_title('Model Comparison')
        ax6.tick_params(axis='x', rotation=45)
        
        # Color bars by performance
        for bar, r2 in zip(bars, r2_scores):
            bar.set_color(plt.cm.viridis(r2))
            
        plots_dir = "../plots"
        os.makedirs(plots_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'tully_fisher_symbolic_regression_results.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 7. Top 5 models: predicted vs true plots with equations
        # Sort models by R² and select top 5
        top_models = sorted(self.results.items(), key=lambda x: x[1]['r2'], reverse=True)[:5]
        n_top = len(top_models)
        if n_top > 0:
            fig2, axes = plt.subplots(1, n_top, figsize=(6*n_top, 6))
            if n_top == 1:
                axes = [axes]
            for i, (model_name, result) in enumerate(top_models):
                model = result['model']
                feature_names = getattr(model, 'feature_names_in_', None)
                if feature_names is not None:
                    features_for_pred = self.features.loc[:, feature_names]
                else:
                    features_for_pred = self.features
                predictions = model.predict(features_for_pred)
                ax = axes[i]
                ax.scatter(self.target, predictions, alpha=0.6, s=20)
                min_val = min(self.target.min(), predictions.min())
                max_val = max(self.target.max(), predictions.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                ax.set_xlabel('True Absolute Magnitude')
                ax.set_ylabel('Predicted Absolute Magnitude')
                ax.set_title(f"{model_name}")
                # Get LaTeX equation for this model
                latex_table = model.latex_table() if callable(getattr(model, 'latex_table', None)) else model.latex_table
                if hasattr(latex_table, 'iloc'):
                    best_eq = latex_table.iloc[0]
                    latex_eq = best_eq['latex_format']
                else:
                    latex_eq = str(latex_table)
                # Display equation in the plot (top left)
                # (Removed as per user request)
                # ax.text(0.02, 0.98, f"${latex_eq}$", transform=ax.transAxes, fontsize=10, va='top', ha='left', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            plt.tight_layout()
            plots_dir = "../plots"
            os.makedirs(plots_dir, exist_ok=True)
            plt.savefig(os.path.join(plots_dir, 'tully_fisher_top5_pred_vs_true.png'), dpi=300, bbox_inches='tight')
            plt.show()
        
    def save_results(self, filename='tully_fisher_symbolic_regression_results.csv'):
        """Save detailed results to CSV"""
        models_dir = "../models"
        os.makedirs(models_dir, exist_ok=True)
        filepath = os.path.join(models_dir, filename)
        if not self.results:
            print("No results to save.")
            return
            
        all_results = []
        
        for model_name, result in self.results.items():
            model = result['model']
            
            try:
                # Try to get equations table from PySR model
                if hasattr(model, 'equations_'):
                    equations_df = model.equations_
                    for idx, row in equations_df.iterrows():
                        # Calculate individual equation metrics if possible
                        try:
                            # Get features for this model
                            feature_names = getattr(model, 'feature_names_in_', None)
                            if feature_names is not None:
                                features_for_pred = self.features.loc[:, feature_names]
                            else:
                                features_for_pred = self.features
                            
                            # Predict using the specific equation (not available in PySR directly)
                            # So we'll use the loss and score from the equations table
                            equation_loss = row.get('loss', None)
                            equation_score = row.get('score', None)
                            
                            # Convert loss to approximate R²-like metric
                            if equation_loss is not None:
                                # Approximate R² = 1 - (loss / var(target))
                                target_var = np.var(self.target)
                                approx_r2 = max(0, 1 - (equation_loss / target_var)) if target_var > 0 else 0
                            else:
                                approx_r2 = result['r2']  # Fallback to overall R²
                                
                            # Use sqrt of loss as RMSE approximation
                            if equation_loss is not None:
                                approx_rmse = np.sqrt(equation_loss)
                            else:
                                approx_rmse = result['rmse']  # Fallback to overall RMSE
                        
                        except Exception as e:
                            # Fallback to overall metrics
                            approx_r2 = result['r2']
                            approx_rmse = result['rmse']
                            equation_score = row.get('score', 0)
                        
                        # Create LaTeX format from equation
                        try:
                            equation_str = str(row.get('equation', 'N/A'))
                            latex_eq = f"${equation_str}$"
                        except:
                            latex_eq = f"$N/A$"
                            
                        all_results.append({
                            'model_name': model_name,
                            'rank': idx + 1,
                            'equation': str(row.get('equation', 'N/A')),
                            'latex': latex_eq,
                            'complexity': row.get('complexity', None),
                            'loss': row.get('loss', None),
                            'score': row.get('score', equation_score),
                            'r2_individual': approx_r2,
                            'rmse_individual': approx_rmse,
                            'r2_overall': result['r2'],
                            'rmse_overall': result['rmse']
                        })
                else:
                    # Fallback: single equation from model
                    try:
                        equation_str = str(model.sympy())
                        latex_eq = f"${equation_str}$"
                    except:
                        equation_str = "Complex equation"
                        latex_eq = "$Complex equation$"
                        
                    all_results.append({
                        'model_name': model_name,
                        'rank': 1,
                        'equation': equation_str,
                        'latex': latex_eq,
                        'complexity': None,
                        'loss': None,
                        'score': None,
                        'r2_individual': result['r2'],
                        'rmse_individual': result['rmse'],
                        'r2_overall': result['r2'],
                        'rmse_overall': result['rmse']
                    })
                    
            except Exception as e:
                print(f"Warning: Error processing {model_name}: {e}")
                # Minimal fallback
                all_results.append({
                    'model_name': model_name,
                    'rank': 1,
                    'equation': 'Error extracting equation',
                    'latex': '$Error extracting equation$',
                    'complexity': None,
                    'loss': None,
                    'score': None,
                    'r2_individual': result['r2'],
                    'rmse_individual': result['rmse'],
                    'r2_overall': result['r2'],
                    'rmse_overall': result['rmse']
                })
                
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")
        print(f"Saved {len(all_results)} equations from {len(self.results)} models")
        
    def export_equations_pdf(self, csv_file='tully_fisher_symbolic_regression_results.csv', pdf_file='tully_fisher_equations.pdf'):
        """Export all LaTeX equations from the CSV to a PDF file automatically."""
        models_dir = "../models"
        os.makedirs(models_dir, exist_ok=True)
        csv_filepath = os.path.join(models_dir, csv_file)
        pdf_filepath = os.path.join(models_dir, pdf_file)
        
        try:
            df = pd.read_csv(csv_filepath)
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return
            
        tex_file = pdf_filepath.replace('.pdf', '.tex')
        
        try:
            with open(tex_file, 'w') as f:
                f.write(r"""\documentclass{article}
\usepackage{amsmath}
\usepackage{geometry}
\geometry{margin=1in}
\begin{document}
\section*{Tully-Fisher Symbolic Regression Equations}

""")
                for idx, row in df.iterrows():
                    # Clean model name for LaTeX
                    model_name = str(row['model_name']).replace('_', '\\_')
                    rank = row['rank']
                    
                    # Use individual metrics if available, otherwise fall back to overall
                    if 'r2_individual' in row and not pd.isna(row['r2_individual']):
                        r2 = row['r2_individual']
                        rmse = row['rmse_individual']
                        metric_label = "Individual"
                    else:
                        r2 = row['r2_overall']
                        rmse = row['rmse_overall']
                        metric_label = "Overall"
                    
                    f.write(f"\\subsection*{{Model: {model_name} (Rank {rank})}}\n")
                    f.write(f"{metric_label} R$^2$: {r2:.4f}, RMSE: {rmse:.4f}\\\\\n")
                    
                    if not pd.isna(row['complexity']) and row['complexity'] is not None:
                        f.write(f"Complexity: {int(row['complexity'])}\\\\\n")
                    
                    if 'score' in row and not pd.isna(row['score']) and row['score'] is not None:
                        f.write(f"Score: {row['score']:.6f}\\\\\n")
                    
                    # Clean the equation for LaTeX
                    equation = str(row.get('equation', 'N/A'))
                    if equation and equation != 'N/A' and equation != 'nan':
                        # Clean equation for LaTeX
                        equation_clean = equation.replace('_', '\\_').replace('&', '\\&').replace('%', '\\%')
                        f.write("\\begin{equation*}\n")
                        f.write(f"y = {equation_clean}")
                        f.write("\n\\end{equation*}\n\n")
                    else:
                        f.write("\\textit{Equation not available}\n\n")
                        
                f.write("\\end{document}\n")
                
            print(f"LaTeX file created: {tex_file}")
            
            # Compile to PDF
            try:
                result = subprocess.run(['pdflatex', '-interaction=nonstopmode', tex_file], 
                                      capture_output=True, text=True, check=True)
                print(f"PDF generated successfully: {pdf_filepath}")
            except subprocess.CalledProcessError as e:
                print(f"Error running pdflatex: {e}")
                print(f"pdflatex stdout: {e.stdout}")
                print(f"pdflatex stderr: {e.stderr}")
                print("You may need to install a LaTeX distribution (e.g., TeX Live or MacTeX).")
            except FileNotFoundError:
                print("pdflatex not found. Please install a LaTeX distribution (e.g., TeX Live or MacTeX).")
                
            # Clean up auxiliary files
            for ext in ['aux', 'log', 'out']:
                aux_file = tex_file.replace('.tex', f'.{ext}')
                if os.path.exists(aux_file):
                    try:
                        os.remove(aux_file)
                    except:
                        pass
                        
        except Exception as e:
            print(f"Error creating LaTeX file: {e}")
            
    def run_full_analysis(self, data_file):
        """Run the complete symbolic regression analysis pipeline"""
        print("Starting Tully-Fisher Symbolic Regression Analysis")
        print("="*60)
        
        # Load and prepare data
        self.load_data(data_file)
        if self.data is None:
            return
            
        self.prepare_features()
        if self.features is None:
            return
            
        # Run comprehensive search
        self.run_comprehensive_search()
        
        # Analyze results
        self.analyze_results()
        
        # Create visualizations
        self.visualize_results()
        
        # Save results
        self.save_results()
        # Export equations PDF automatically
        self.export_equations_pdf()
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("Check the generated plots, CSV file, and PDF for detailed results.")


def main():
    """Main execution function"""
    
    # Example usage
    if PYSR_AVAILABLE:
        # Initialize the symbolic regression system
        tf_sr = TullyFisherSymbolicRegression(random_state=42)
        
        # Example: Run with a data file
        # tf_sr.run_full_analysis('your_tully_fisher_data.csv')
        
        print("Tully-Fisher Symbolic Regression System Ready!")
        print("Usage:")
        print("  tf_sr = TullyFisherSymbolicRegression()")
        print("  tf_sr.run_full_analysis('your_data.csv')")
        print("\nOr run step by step:")
        print("  tf_sr.load_data('your_data.csv')")
        print("  tf_sr.prepare_features()")
        print("  tf_sr.run_comprehensive_search()")
        print("  tf_sr.analyze_results()")
        print("  tf_sr.visualize_results()")
        
    else:
        print("PySR not available. Install with:")
        print("  pip install pysr")
        print("  python -m pysr install")


if __name__ == "__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser(description="Run PySR Tully-Fisher symbolic regression analysis.")
    parser.add_argument('--data', type=str, required=False, help='Path to the input CSV data file')
    args = parser.parse_args()

    # Default data file path
    default_data_path = os.path.join(os.path.dirname(__file__), "../input/tully_fisher_simulated_dataset.csv")
    data_file = args.data if args.data else default_data_path

    print("=== PySR Tully-Fisher Symbolic Regression Analysis ===")
    tf_sr = TullyFisherSymbolicRegression()
    if os.path.exists(data_file):
        tf_sr.run_full_analysis(data_file)
    else:
        print(f"Data file not found: {data_file}\nPlease provide a valid CSV file with --data or place your data at the default location.")
