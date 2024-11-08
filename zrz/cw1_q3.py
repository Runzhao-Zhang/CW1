import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
import time

data = scipy.io.loadmat('face.mat')
face_data = data['X'] # 2576 x 520
face_identities = data['l'] # 520 x 1

X_train, X_test, y_train, y_test = train_test_split(face_data.T, face_identities.T, test_size=0.2, random_state=42)

# Total scatter matrix rank
S_t = np.cov(X_train.T)
print(f"Total scatter matrix rank: {np.linalg.matrix_rank(S_t)}")

M_pca_values = [50, 100, 150, 200, 250, 300, 350, 400]
M_lda_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
results = {}
times = {}

for M_pca in M_pca_values:
    # PCA
    pca = PCA(n_components=M_pca)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    for M_lda in M_lda_values:
        # LDA
        lda = LinearDiscriminantAnalysis(n_components=M_lda)
        X_train_lda = lda.fit_transform(X_train_pca, y_train.ravel())
        X_test_lda = lda.transform(X_test_pca)
        
        # KNN
        knn = KNeighborsClassifier(n_neighbors=1)
        start_time = time.time()
        knn.fit(X_train_lda, y_train.ravel())
        y_pred = knn.predict(X_test_lda)
        end_time = time.time()
        
        # accuracy
        acc = accuracy_score(y_test.ravel(), y_pred)

        results[(M_pca, M_lda)] = acc
        times[(M_pca, M_lda)] = end_time - start_time
        # print(f"M_pca={M_pca}, M_lda={M_lda}: Accuracy={acc:.3f}, Time={end_time-start_time:.2f}s")

# plot 3D surface
fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title('Accuracy')
ax1.set_xlabel('M_pca')
ax1.set_ylabel('M_lda')
ax1.set_zlabel('Accuracy')
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_title('Time')
ax2.set_xlabel('M_pca')
ax2.set_ylabel('M_lda')
ax2.set_zlabel('Time')

X, Y = np.meshgrid(M_pca_values, M_lda_values)
Z_acc = np.array([[results[(x, y)] for x in M_pca_values] for y in M_lda_values])
Z_time = np.array([[times[(x, y)] for x in M_pca_values] for y in M_lda_values])

ax1.plot_surface(X, Y, Z_acc, cmap='viridis') 
ax2.plot_surface(X, Y, Z_time, cmap='viridis')
plt.show()

# plot confusion matrix
best_M_pca, best_M_lda = max(results, key=results.get)
print(f"Best M_pca={best_M_pca}, M_lda={best_M_lda}, Accuracy={results[(best_M_pca, best_M_lda)]:.3f}")
pca = PCA(n_components=best_M_pca)
lda = LinearDiscriminantAnalysis(n_components=best_M_lda)
knn = KNeighborsClassifier(n_neighbors=1)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
X_train_lda = lda.fit_transform(X_train_pca, y_train.ravel())
X_test_lda = lda.transform(X_test_pca)
knn.fit(X_train_lda, y_train.ravel())
y_pred = knn.predict(X_test_lda)

cm = confusion_matrix(y_test.ravel(), y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, cmap='Blues')
plt.title(f'Confusion Matrix (M_pca={best_M_pca}, M_lda={best_M_lda})')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# plot success and failure cases
success_idx = np.where(y_pred == y_test.ravel())[0][0]
failure_idx = np.where(y_pred != y_test.ravel())[0][0]

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(X_test[success_idx].reshape(46, 56).T, cmap='gray')
plt.title(f'Success\nTrue: {y_test.ravel()[success_idx]}\nPred: {y_pred[success_idx]}')
plt.subplot(122)
plt.imshow(X_test[failure_idx].reshape(46, 56).T, cmap='gray')
plt.title(f'Failure\nTrue: {y_test.ravel()[failure_idx]}\nPred: {y_pred[failure_idx]}')
plt.show()
