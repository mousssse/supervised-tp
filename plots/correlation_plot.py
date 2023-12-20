import matplotlib.pyplot as plt
import numpy as np

# Features et corrélations
feature_names = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'SEX', 'RAC1P']
correlations_train = np.array([0.0609, 0.3492, -0.2657, -0.3368, -0.0832, -0.2189, 0.3418, -0.1174, -0.0970])

# Trier les indices en fonction des corrélations
correlation_train_indices = np.argsort(np.abs(correlations_train))[::-1]

# Créer une figure
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(np.array(feature_names)[correlation_train_indices], np.abs(correlations_train[correlation_train_indices]))

# Ajuster les étiquettes et les titres
ax.set_xlabel('Corrélations avec la prédiction d\'entraînement')
ax.set_title('Corrélations entre les features et la prédiction d\'entraînement')

# Afficher le graphique
plt.savefig('plots/pdf/correlations.pdf')

correlations_adaboost = np.array([0.0774, 0.4595, -0.3225, -0.4587, -0.1149, -0.2660, 0.4094, -0.1373, -0.1275])
correlations_gb = np.array([0.0750, 0.4638, -0.3240, -0.4654, -0.1186, -0.2693, 0.3848, -0.1179, -0.1199])
correlations_rf = np.array([0.0772, 0.4688, -0.3177, -0.4849, -0.1244, -0.2767, 0.4012, -0.1151, -0.1218])
correlations_svm = np.array([0.0804, 0.4663, -0.3579, -0.5062, -0.1258, -0.2907, 0.3958, -0.1330, -0.1228])

# Trier les indices en fonction des corrélations
adaboost_correlation_indices = np.argsort(np.abs(correlations_adaboost))[::-1]
gb_correlation_indices = np.argsort(np.abs(correlations_gb))[::-1]
rf_correlation_indices = np.argsort(np.abs(correlations_rf))[::-1]
svm_correlation_indices = np.argsort(np.abs(correlations_svm))[::-1]

# Créer une figure avec plusieurs sous-graphiques
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Corrélations entre les features et la prédiction pour chaque modèle', fontsize=16)

# Plot pour AdaBoost
axs[0, 0].bar(np.array(feature_names)[adaboost_correlation_indices], np.abs(correlations_adaboost[adaboost_correlation_indices]), color='darkorange')
axs[0, 0].set_title('AdaBoost')

# Plot pour Gradient Boost
axs[0, 1].bar(np.array(feature_names)[gb_correlation_indices], np.abs(correlations_gb[gb_correlation_indices]), color='mediumorchid')
axs[0, 1].set_title('Gradient Boost')

# Plot pour Random Forest
axs[1, 0].bar(np.array(feature_names)[rf_correlation_indices], np.abs(correlations_rf[rf_correlation_indices]), color='forestgreen')
axs[1, 0].set_title('Random Forest')

# Plot pour SVM
axs[1, 1].bar(np.array(feature_names)[svm_correlation_indices], np.abs(correlations_svm[svm_correlation_indices]), color='skyblue')
axs[1, 1].set_title('SVM')

# Afficher le graphique
plt.savefig('plots/pdf/models_correlations.pdf')
