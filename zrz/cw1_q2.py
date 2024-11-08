import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA, IncrementalPCA
import time

data = scipy.io.loadmat('face.mat')
face_data = data['X'] # 2576 x 520
face_identities = data['l'] # 520 x 1

X_train, X_test, y_train, y_test = train_test_split(face_data.T, face_identities.T, test_size=0.2, random_state=42)
# Split the training data into 4 subsets
n_samples_per_subset = len(X_train) // 4
subsets = []
for i in range(4):
    start_idx = i * n_samples_per_subset
    end_idx = (i + 1) * n_samples_per_subset
    subsets.append((X_train[start_idx:end_idx], y_train[start_idx:end_idx]))
    # print(subsets[i][0].shape, subsets[i][1].shape)

# Three PCA methods to compare
def compare_pca_methods(n_components=100):
    results = {
        'training_time': [],
        'reconstruction_error': [],
        'recognition_accuracy': []
    }
    
    # Incremental PCA
    start_time = time.time()
    ipca = IncrementalPCA(n_components=n_components) 
    for i in range(len(subsets)):
        current_data = np.vstack([subsets[j][0] for j in range(i + 1)])
        # print(current_data.shape)
        ipca.partial_fit(current_data)
    
    ipca_time = time.time() - start_time
    
    # batch PCA
    start_time = time.time()
    pca = PCA(n_components=n_components) 
    pca.fit(X_train)
    batch_time = time.time() - start_time
    
    # Subsample PCA
    start_time = time.time()
    single_pca = PCA(n_components=n_components)
    single_pca.fit(subsets[0][0])
    single_time = time.time() - start_time
    
    # reconstruction error
    ipca_error = np.mean(np.sum((X_train - ipca.inverse_transform(ipca.transform(X_train))) ** 2, axis=1))
    batch_error = np.mean(np.sum((X_train - pca.inverse_transform(pca.transform(X_train))) ** 2, axis=1))
    single_error = np.mean(np.sum((X_train - single_pca.inverse_transform(single_pca.transform(X_train))) ** 2, axis=1))
    
    # accuracy
    def get_recognition_accuracy(pca_model):
        X_train_transformed = pca_model.transform(X_train)
        X_test_transformed = pca_model.transform(X_test)
        
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_train_transformed, y_train.ravel())
        y_pred = knn.predict(X_test_transformed)
        return accuracy_score(y_test.ravel(), y_pred)
    
    ipca_acc = get_recognition_accuracy(ipca)
    batch_acc = get_recognition_accuracy(pca)
    single_acc = get_recognition_accuracy(single_pca)
    
    return {
        'IPCA': (ipca_time, ipca_error, ipca_acc),
        'Batch PCA': (batch_time, batch_error, batch_acc),
        'Single PCA': (single_time, single_error, single_acc)
    }

n_components_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
results_all = {}

for n in n_components_list:
    results = compare_pca_methods(n_components=n)
    results_all[n] = results

# Plot results
metrics = ['Training Time (s)', 'Reconstruction Error', 'Recognition Accuracy']
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, metric in enumerate(metrics):
    for method in ['IPCA', 'Batch PCA', 'Single PCA']:
        values = [results_all[n][method][i] for n in n_components_list]
        axes[i].plot(n_components_list, values, marker='o', label=method)
    
    axes[i].set_xlabel('Number of Components')
    axes[i].set_ylabel(metric)
    axes[i].legend()

plt.tight_layout()
plt.show()


