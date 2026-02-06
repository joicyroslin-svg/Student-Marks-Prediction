import pandas as pd          # For working with data in tables
import numpy as np           # For numbers and random data
import matplotlib.pyplot as plt  # For creating graphs
from sklearn.model_selection import train_test_split    # To split data into training and testing
from sklearn.linear_model import LinearRegression       # The machine learning model
from sklearn.metrics import mean_squared_error, r2_score  # To check how good our model is
import seaborn as sns        # For making nicer graphs

print("="*50)
print("STUDENT MARKS PREDICTION PROJECT")
print("="*50)

# ===== STEP 1: CREATE SAMPLE DATA =====
print("\n[STEP 1] Creating 100 students with their data...")

np.random.seed(42)  # This makes the random numbers same every time we run
n_samples = 100     # Number of students

# Create a dictionary with student information
data = {
    'study_hours': np.random.uniform(1, 10, n_samples),      # Random hours (1-10)
    'attendance': np.random.uniform(50, 100, n_samples),     # Random attendance (50-100%)
    'previous_score': np.random.uniform(40, 90, n_samples)   # Random previous marks (40-90)
}

# Convert dictionary into a table (DataFrame)
df = pd.DataFrame(data)

# ===== STEP 2: CREATE THE MARKS COLUMN =====
# Formula: marks = (5 × study_hours) + (0.3 × attendance) + (0.5 × previous_score) + random noise
# This means: studying more = bigger marks increase, attendance helps, previous score helps
df['marks'] = (5 * df['study_hours'] + 
               0.3 * df['attendance'] + 
               0.5 * df['previous_score'] + 
               np.random.normal(0, 5, n_samples))  # Add some randomness to be realistic

print(f"[OK] Created data for {n_samples} students")
print(f"\nFirst 5 students:")
print(df.head())

# ===== STEP 3: SPLIT DATA INTO TRAINING AND TESTING =====
print("\n[STEP 3] Splitting data...")

# X = Input features (study_hours, attendance, previous_score)
# y = Output we want to predict (marks)
X = df[['study_hours', 'attendance', 'previous_score']]  # Features (input)
y = df['marks']  # Target (what we want to predict)

# Split: 80% for training the model, 20% for testing
# random_state=42 means the split is always the same for consistency
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"[OK] Training data: {len(X_train)} students")
print(f"[OK] Testing data: {len(X_test)} students")

# ===== STEP 4: TRAIN THE MODEL =====
print("\n[STEP 4] Training the machine learning model...")

# Create a Linear Regression model (finds best fitting line through data)
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

print("[OK] Model trained successfully!")

# ===== STEP 5: MAKE PREDICTIONS =====
print("\n[STEP 5] Making predictions on test data...")

# Use the trained model to predict marks for the test students
y_pred = model.predict(X_test)

print(f"[OK] Predicted marks for {len(y_test)} test students")

# ===== STEP 6: CHECK HOW GOOD OUR MODEL IS =====
print("\n[STEP 6] Evaluating model performance...")

# R² Score: How close our predictions are (1.0 = perfect, 0.0 = terrible)
r2 = r2_score(y_test, y_pred)

# MSE: Average error in predictions (lower is better)
mse = mean_squared_error(y_test, y_pred)

print(f"\n{'='*50}")
print("MODEL RESULTS:")
print(f"{'='*50}")
print(f"R² Score: {r2:.4f} (Good! Close to 1.0)")
print(f"  → This means {r2*100:.1f}% of variation is explained by our model")
print(f"\nMean Squared Error: {mse:.2f}")
print(f"  → Average prediction error")

print(f"\n{'='*50}")
print("HOW MUCH EACH FACTOR AFFECTS MARKS:")
print(f"{'='*50}")

# Show the coefficients (how much each input affects the output)
coefficients = dict(zip(X.columns, model.coef_))
for feature, coefficient in coefficients.items():
    print(f"{feature:20s}: {int(coefficient)}")
    
print(f"\n→ Study Hours has the BIGGEST effect on marks!")
print(f"{'='*50}\n")

# ===== STEP 7: CREATE VISUALIZATIONS =====
print("[STEP 7] Creating graphs...")

# Create a figure with 2 subplots side by side
fig = plt.figure(figsize=(14, 5))
fig.patch.set_facecolor('#E8F4F8')  # Beautiful light blue gradient
plt.suptitle('Figure 1: Machine Learning Model Performance', fontsize=14, fontweight='bold', y=0.98, color='#1a1a1a')

# SUBPLOT 1: Actual vs Predicted Marks (Scatter Plot)
ax1 = plt.subplot(1, 2, 1)
ax1.set_facecolor('#ffffff')  # White background
plt.scatter(y_test, y_pred, alpha=0.7, color='#2E86AB', s=100, edgecolors='black', linewidth=0.5)  # Nice blue
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Marks', fontsize=12, fontweight='bold')
plt.ylabel('Predicted Marks', fontsize=12, fontweight='bold')
plt.title('Actual vs Predicted Marks', fontsize=13, fontweight='bold', pad=10)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# SUBPLOT 2: Feature Correlation (Heatmap)
ax2 = plt.subplot(1, 2, 2)
ax2.set_facecolor('#ffffff')
sns.heatmap(df.corr(), annot=True, cmap='RdYlGn', cbar_kws={'label': 'Correlation'}, 
            square=True, linewidths=1, linecolor='white', fmt='.2f')
plt.title('Feature Correlation Matrix', fontsize=13, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig('marks_prediction_plot.png', dpi=100)
print("[OK] Plot saved as 'marks_prediction_plot.png'")
plt.show()
print("\n[OK] PROJECT COMPLETE!")