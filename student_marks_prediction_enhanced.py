"""
STUDENT MARKS PREDICTION PROJECT - ENHANCED VERSION
====================================================
This project demonstrates Machine Learning basics using Python.
It predicts student exam marks based on study hours, attendance, and previous scores.

Features:
- Data generation and analysis
- Linear Regression model training
- Model evaluation and visualization
- Custom student predictions
- Export results to CSV files

Author: JOICY ROSLIN
Date: February 2026
"""

import pandas as pd          
import numpy as np           
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split    
from sklearn.linear_model import LinearRegression       
from sklearn.metrics import mean_squared_error, r2_score  
import seaborn as sns        

def create_sample_data(n_samples=100):
    """
    Generate sample student data with marks
    
    Parameters:
    - n_samples: Number of students to generate (default: 100)
    
    Returns:
    - DataFrame with student data
    """
    np.random.seed(42)  # For reproducibility
    
    # Create random data for each feature
    data = {
        'study_hours': np.random.uniform(1, 10, n_samples),
        'attendance': np.random.uniform(50, 100, n_samples),
        'previous_score': np.random.uniform(40, 90, n_samples)
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Calculate marks using a formula
    # Marks = 5*study_hours + 0.3*attendance + 0.5*previous_score + noise
    df['marks'] = (5 * df['study_hours'] + 
                   0.3 * df['attendance'] + 
                   0.5 * df['previous_score'] + 
                   np.random.normal(0, 5, n_samples))
    
    return df

def train_and_evaluate_model(df):
    """
    Train the machine learning model and evaluate it
    
    Parameters:
    - df: DataFrame with student data
    
    Returns:
    - model: Trained Linear Regression model
    - X_test, y_test, y_pred: Test data and predictions
    - r2, mse: Model evaluation metrics
    """
    # Prepare features and target
    X = df[['study_hours', 'attendance', 'previous_score']]
    y = df['marks']
    
    # Split data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate model
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    return model, X_test, y_test, y_pred, r2, mse

def save_results(model, y_test, y_pred, r2, mse):
    """
    Save model results and predictions to CSV files
    
    Parameters:
    - model: Trained model
    - y_test, y_pred: Test data and predictions
    - r2, mse: Model metrics
    """
    # Save model results
    results_data = {
        'Model': ['Linear Regression'],
        'R-Square Score': [r2],
        'Mean Squared Error': [mse],
        'Study Hours Coefficient': [model.coef_[0]],
        'Attendance Coefficient': [model.coef_[1]],
        'Previous Score Coefficient': [model.coef_[2]]
    }
    results_df = pd.DataFrame(results_data)
    results_df.to_csv('model_results.csv', index=False)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'Actual Marks': y_test.values,
        'Predicted Marks': y_pred,
        'Error': y_test.values - y_pred
    })
    predictions_df.to_csv('predictions.csv', index=False)

def predict_custom_students(model):
    """
    Predict marks for custom students
    
    Parameters:
    - model: Trained model
    """
    # Create example custom students
    custom_students = pd.DataFrame({
        'study_hours': [8, 6, 9],
        'attendance': [85, 70, 95],
        'previous_score': [75, 60, 88]
    })
    
    # Make predictions
    custom_predictions = model.predict(custom_students)
    
    print("\n" + "="*60)
    print("CUSTOM STUDENT PREDICTIONS")
    print("="*60)
    
    for i, (idx, row) in enumerate(custom_students.iterrows()):
        print(f"\nStudent {i+1}:")
        print(f"  Study Hours: {row['study_hours']:.1f} hours")
        print(f"  Attendance: {row['attendance']:.1f}%")
        print(f"  Previous Score: {row['previous_score']:.1f}")
        print(f"  PREDICTED MARKS: {custom_predictions[i]:.2f} ⭐")
    
    print("\n" + "="*60)

def create_visualizations(df, y_test, y_pred):
    """
    Create beautiful visualizations
    
    Parameters:
    - df: Original DataFrame
    - y_test, y_pred: Test data and predictions
    """
    # Create figure with 2 subplots
    fig = plt.figure(figsize=(14, 5))
    fig.patch.set_facecolor('#E8F4F8')  # Light blue background
    plt.suptitle('Figure 1: Machine Learning Model Performance', 
                 fontsize=14, fontweight='bold', y=0.98, color='#1a1a1a')

    # SUBPLOT 1: Actual vs Predicted Marks
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_facecolor('#ffffff')
    plt.scatter(y_test, y_pred, alpha=0.7, color='#2E86AB', s=100, 
                edgecolors='black', linewidth=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Actual Marks', fontsize=12, fontweight='bold')
    plt.ylabel('Predicted Marks', fontsize=12, fontweight='bold')
    plt.title('Actual vs Predicted Marks', fontsize=13, fontweight='bold', pad=10)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # SUBPLOT 2: Feature Correlation Matrix
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_facecolor('#ffffff')
    sns.heatmap(df.corr(), annot=True, cmap='RdYlGn', 
                cbar_kws={'label': 'Correlation'}, 
                square=True, linewidths=1, linecolor='white', fmt='.2f')
    plt.title('Feature Correlation Matrix', fontsize=13, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig('marks_prediction_plot.png', dpi=100)
    plt.show()

def print_results(model, r2, mse):
    """
    Print detailed model results
    
    Parameters:
    - model: Trained model
    - r2, mse: Model metrics
    """
    print(f"\n{'='*50}")
    print("MODEL RESULTS:")
    print(f"{'='*50}")
    print(f"R² Score: {r2:.4f} (Excellent! Close to 1.0)")
    print(f"  -> {r2*100:.1f}% of variation is explained by our model")
    print(f"\nMean Squared Error: {mse:.2f}")
    print(f"  -> Average prediction error")

    print(f"\n{'='*50}")
    print("HOW MUCH EACH FACTOR AFFECTS MARKS:")
    print(f"{'='*50}")

    coefficients = dict(zip(['study_hours', 'attendance', 'previous_score'], model.coef_))
    for feature, coefficient in coefficients.items():
        print(f"{feature:20s}: {int(coefficient):3d}")
        
    print(f"\n-> Study Hours has the BIGGEST effect on marks!")
    print(f"{'='*50}\n")

def main():
    """
    Main function - orchestrates the entire project workflow
    """
    print("="*60)
    print("STUDENT MARKS PREDICTION - MACHINE LEARNING PROJECT")
    print("="*60)

    # ===== STEP 1: CREATE DATA =====
    print("\n[STEP 1] Creating 100 students with their data...")
    df = create_sample_data(n_samples=100)
    print(f"[OK] Created {len(df)} students")
    print(f"\nFirst 5 students:")
    print(df.head())

    # ===== STEP 2: SPLIT AND TRAIN =====
    print("\n[STEP 2] Splitting data and training model...")
    model, X_test, y_test, y_pred, r2, mse = train_and_evaluate_model(df)
    print(f"[OK] Training data: 80 students")
    print(f"[OK] Testing data: 20 students")
    print(f"[OK] Model trained successfully!")

    # ===== STEP 3: EVALUATE =====
    print("\n[STEP 3] Evaluating model performance...")
    print_results(model, r2, mse)

    # ===== STEP 4: SAVE RESULTS =====
    print("[STEP 4] Saving results to CSV files...")
    save_results(model, y_test, y_pred, r2, mse)
    print("[OK] Results saved to 'model_results.csv'")
    print("[OK] Predictions saved to 'predictions.csv'")

    # ===== STEP 5: CUSTOM PREDICTIONS =====
    print("\n[STEP 5] Making custom student predictions...")
    predict_custom_students(model)

    # ===== STEP 6: VISUALIZATIONS =====
    print("[STEP 6] Creating beautiful visualizations...")
    create_visualizations(df, y_test, y_pred)
    print("[OK] Graph saved as 'marks_prediction_plot.png'")

    print("\n" + "="*60)
    print("PROJECT COMPLETE! ✓")
    print("="*60)
    print("\nFiles created:")
    print("  ✓ marks_prediction_plot.png")
    print("  ✓ model_results.csv")
    print("  ✓ predictions.csv")
    print("\n" + "="*60)

# ===== RUN THE PROJECT =====
if __name__ == "__main__":
    main()
