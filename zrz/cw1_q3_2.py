import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
from scipy.stats import mode
import time

data = scipy.io.loadmat('face.mat')
face_data = data['X'] # 2576 x 520
face_identities = data['l'] # 520 x 1

X_train, X_test, y_train, y_test = train_test_split(face_data.T, face_identities.T, test_size=0.2, random_state=42)

class PCALDAEnsemble:
    def __init__(self, n_models=10, pca_components=150, lda_components=45, 
                 feature_fraction=0.8, sample_fraction=0.8):
        self.n_models = n_models
        self.pca_components = pca_components
        self.lda_components = lda_components
        self.feature_fraction = feature_fraction
        self.sample_fraction = sample_fraction
        self.models = []
        
    def _create_base_model(self):
        return {
            'pca': PCA(n_components=self.pca_components),
            'lda': LinearDiscriminantAnalysis(n_components=self.lda_components),
            'knn': KNeighborsClassifier(n_neighbors=1)
        }
    
    def fit(self, X, y):
        n_features = X.shape[1]
        n_feature_subset = int(n_features * self.feature_fraction) # how many features to select
        
        for _ in range(self.n_models):
            # random choice of features and no repeat
            feature_indices = np.random.choice(n_features, n_feature_subset, replace=False)
            
            # Bagging
            X_bag, y_bag = resample(X, y, random_state=np.random.randint(0, 1000), 
                                    n_samples=int(self.sample_fraction * X.shape[0]))
            
            # Create base model
            model = self._create_base_model()
            X_pca = model['pca'].fit_transform(X_bag[:, feature_indices])
            X_lda = model['lda'].fit_transform(X_pca, y_bag.ravel())
            model['knn'].fit(X_lda, y_bag.ravel())
            
            self.models.append({
                'model': model,
                'feature_indices': feature_indices
            })
    
    def predict(self, X, fusion_rule='majority'):
        predictions = []
        
        for model_info in self.models:
            model = model_info['model']
            feature_indices = model_info['feature_indices']
            
            X_pca = model['pca'].transform(X[:, feature_indices])
            X_lda = model['lda'].transform(X_pca)
            pred = model['knn'].predict(X_lda)
            predictions.append(pred)
            
        predictions = np.array(predictions)
        
        if fusion_rule == 'majority':
            return mode(predictions, axis=0, keepdims=True)[0].ravel()
        elif fusion_rule == 'mean':
            return np.round(np.mean(predictions, axis=0)).astype(int)

n_models_list = [5, 10, 15] # number of base models
feature_fractions = [0.5, 0.7, 0.9] # Randomisation in feature selection
sample_fractions = [0.5, 0.7, 0.9] # Randomisation in sample selection
fusion_rules = ['majority', 'mean'] # Fusion rules

results = {}

for n_models in n_models_list:
    for feat_frac in feature_fractions:
        for sample_frac in sample_fractions:
            for fusion_rule in fusion_rules:
                ensemble = PCALDAEnsemble(
                    n_models=n_models,
                    feature_fraction=feat_frac,
                    sample_fraction=sample_frac
                )
                
                start_time = time.time()
                ensemble.fit(X_train, y_train)
                y_pred = ensemble.predict(X_test, fusion_rule=fusion_rule)
                end_time = time.time()
                
                ensemble_acc = accuracy_score(y_test.ravel(), y_pred)
                
                results[(n_models, feat_frac, sample_frac, fusion_rule)] = {
                    'accuracy': ensemble_acc,
                    'time': end_time - start_time
                }
                
                # print(f"Models: {n_models}, Feature Fraction: {feat_frac}, "
                #       f"Sample Fraction: {sample_frac}, Fusion: {fusion_rule}")
                # print(f"Accuracy: {ensemble_acc:.3f}, Time: {end_time-start_time:.2f}s")

# plot 3D surface
fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title('Accuracy')
ax1.set_xlabel('Models')
ax1.set_ylabel('Feature Fraction')
ax1.set_zlabel('Accuracy')
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_title('Time')
ax2.set_xlabel('Models')
ax2.set_ylabel('Feature Fraction')
ax2.set_zlabel('Time')

X, Y = np.meshgrid(n_models_list, feature_fractions)

Z_acc = np.array([[results[(x, y, 0.9, 'majority')]['accuracy'] for x in n_models_list] for y in feature_fractions])
ax1.plot_surface(X, Y, Z_acc, cmap='viridis')
Z_acc = np.array([[results[(x, y, 0.9, 'mean')]['accuracy'] for x in n_models_list] for y in feature_fractions])
ax1.plot_surface(X, Y, Z_acc, cmap='plasma')
Z_time = np.array([[results[(x, y, 0.9, 'majority')]['time'] for x in n_models_list] for y in feature_fractions])
ax2.plot_surface(X, Y, Z_time, cmap='viridis')
Z_time = np.array([[results[(x, y, 0.9, 'mean')]['time'] for x in n_models_list] for y in feature_fractions])
ax2.plot_surface(X, Y, Z_time, cmap='plasma')

plt.show()

best_params = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
best_ensemble = PCALDAEnsemble(
    n_models=best_params[0],
    feature_fraction=best_params[1],
    sample_fraction=best_params[2]
)
print(f"Best Ensemble: {best_params}")
best_ensemble.fit(X_train, y_train)
y_pred = best_ensemble.predict(X_test, fusion_rule=best_params[3])

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test.ravel(), y_pred)
sns.heatmap(cm, cmap='Blues')
plt.title(f'Confusion Matrix (Best Ensemble)\nModels={best_params[0]}, '
          f'Feature Fraction={best_params[1]}, Sample Fraction={best_params[2]}')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()