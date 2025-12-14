import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

df = pd.read_csv("clean_data.csv")

X = df.drop('Churn', axis=1)
y = df['Churn']

# Train/Test Split (70% Train, 30% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=30)

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=30),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(probability=True, random_state=30)
}

results = []
confusion_matrices = {} # Store matrices for later plotting

print("Training Models and Generating ROC Curve..")
plt.figure(figsize=(8, 6)) # Initialize ROC plot area

for name, model in models.items():
    # Train the Model
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    # Calculate and store Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices[name] = cm
    
    # Add results to the list
    results.append([name, acc, prec, rec, f1, auc])

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")

# Finalize and Show ROC Plot
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess') 
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.show()

# Confusion Matrix Visualition
print("\n Generating Confusion Matrix Plots..")
for name, cm in confusion_matrices.items():
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show() 

results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"])
print("\n Model Comparison Table")
print(results_df)

best_model = results_df.sort_values(by="F1 Score", ascending=False).iloc[0]
print(f"\nBest Model (based on F1 Score): {best_model['Model']}")