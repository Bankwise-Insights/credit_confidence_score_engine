from preprocessor import X_train, X_test, y_train, y_test

#2. EDA and Visualization
print("\n\n--- Step 2: Exploratory Data Analysis & Visualization ---")
sns.set_style("whitegrid")
plt.style.use('ggplot')


# Distributions of numerical features
print("Generating distributions for numerical features...")
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(2, 2, i + 1)
    sns.histplot(df[col], kde=True, color='skyblue', bins=20)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# distributions of categorical features
print("Generating distributions for categorical features...")
plt.figure(figsize=(18, 12))
for i, col in enumerate(categorical_cols):
    plt.subplot(3, 3, i + 1)
    sns.countplot(data=df, x=col, palette='viridis')
    plt.title(f'Count of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# Correlation Heatmap for all numerical variables
print("Generating correlation heatmap for numerical features and the target...")
numerical_and_target = df[numerical_cols + [target_col]]
corr_matrix = numerical_and_target.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features and CreditScore')
plt.show()


# Relationships between features and CreditScore
print("Generating visualizations of feature relationships with CreditScore...")


plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(2, 2, i + 1)
    sns.scatterplot(data=df, x=col, y=target_col, alpha=0.6, color='coral')
    plt.title(f'{col} vs. CreditScore')
    plt.xlabel(col)
    plt.ylabel('CreditScore')
plt.tight_layout()
plt.show()


# Box plots for categorical features vs. CreditScore 
print("Generating box plots for categorical features vs. CreditScore...")
plt.figure(figsize=(18, 12))
for i, col in enumerate(categorical_cols):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(data=df, x=col, y=target_col, palette='pastel')
    plt.title(f'{col} vs. CreditScore')
    plt.xlabel(col)
    plt.ylabel('CreditScore')
    plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



# --- 3. Model Training and Evaluation ---
print("\n\n--- Step 3: Model Training and Evaluation ---")

#model performance metrics
performance_metrics = []

# Train and evaluating model
def train_and_evaluate(model, model_name):
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Performance metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    

    performance_metrics.append({
        'Model': model_name,
        'MSE': mse,
        'RMSE': rmse,
        'R2 Score': r2
    })
    # results
    print(f"--- {model_name} Performance ---")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R2) Score: {r2:.4f}")
    
   
    return y_pred


models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
} 

for name, model in models.items():
    y_pred = train_and_evaluate(model, name)
    # actual vs. predicted for each model
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Credit Score')
    plt.ylabel('Predicted Credit Score')
    plt.title(f'Actual vs. Predicted Credit Scores ({name})')
    plt.show()


