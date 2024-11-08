import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import time

data = scipy.io.loadmat('face.mat')
face_data = data['X']  # 2576 x 520
face_identities = data['l']  # 520 x 1

X_train, X_test, y_train, y_test = train_test_split(face_data.T, face_identities.T, test_size=0.2, random_state=42)

n_estimators_values = [10, 50, 100, 150, 200]
max_depth_values = [5, 10, 20, 50, 100]
results = []
time_results = []

# Loop through the parameter combinations and train/test RF models
for n_estimators in n_estimators_values:
    for max_depth in max_depth_values:
        # Create a Random Forest Classifier with the current parameter values
        rf_classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )

        start = time.time()
        rf_classifier.fit(X_train, y_train.ravel())
        end = time.time()

        y_pred = rf_classifier.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        results.append(accuracy)
        time_results.append(end-start)

        print(f'n_estimators={n_estimators}, max_depth={max_depth}' +
              f' Accuracy: {accuracy:.2f}' + f' Time: {end-start:.2f}s')

# Plot the results
results = np.array(results).reshape(len(n_estimators_values), len(max_depth_values))
time_results = np.array(time_results).reshape(len(n_estimators_values), len(max_depth_values))

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.heatmap(results, annot=True, fmt=".2f", ax=ax[0], xticklabels=max_depth_values, yticklabels=n_estimators_values)
ax[0].set_title('Accuracy')
ax[0].set_xlabel('Max Depth')
ax[0].set_ylabel('N Estimators')

sns.heatmap(time_results, annot=True, fmt=".2f", ax=ax[1], xticklabels=max_depth_values, yticklabels=n_estimators_values)
ax[1].set_title('Training Time')
ax[1].set_xlabel('Max Depth')
ax[1].set_ylabel('N Estimators')

plt.tight_layout()
plt.show()

# best parameters
best_params = (n_estimators_values[np.argmax(results) // len(max_depth_values)],
                max_depth_values[np.argmax(results) % len(max_depth_values)])
best_rf = RandomForestClassifier(
    n_estimators=best_params[0],
    max_depth=best_params[1],
    random_state=42
)
best_rf.fit(X_train, y_train.ravel())
y_pred = best_rf.predict(X_test)

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test.ravel(), y_pred)
sns.heatmap(cm, cmap='Blues')

plt.title(f'Confusion Matrix (n_estimators={best_params[0]}, max_depth={best_params[1]})')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# success and failure cases
success_idx = np.where(y_pred == y_test.ravel())[0][0]
failure_idx = np.where(y_pred != y_test.ravel())[0][0]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(X_test[success_idx].reshape(46, 56).T, cmap='gray')
plt.title(f'Success: Predicted={y_pred[success_idx]}, True={y_test.ravel()[success_idx]}')

plt.subplot(1, 2, 2)
plt.imshow(X_test[failure_idx].reshape(46, 56).T, cmap='gray')
plt.title(f'Failure: Predicted={y_pred[failure_idx]}, True={y_test.ravel()[failure_idx]}')

plt.show()


        