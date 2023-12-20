import matplotlib.pyplot as plt
import numpy as np

# Features and importances for each model
feature_names = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'SEX', 'RAC1P']
importances_adaboost = np.array([0.0044206, 0.04172735, 0.00746135, 0.04285167, 0.00367957,
                                 0.03017759, 0.05769771, 0.00981219, 0.00201865])
importances_gb = np.array([0.0039862, 0.03950428, 0.00697585, 0.05575572, 0.00454836,
                           0.02657468, 0.05938418, 0.00884119, 0.00056216])
importances_rf = np.array([0.00421617, 0.0452025, 0.00856011, 0.04783442, 0.00574933,
                           0.03048422, 0.06873643, 0.00958222, 0.00304076])
importances_svm = np.array([0.001942, 0.05309825, 0.02166858, 0.03214514, 0.00462502,
                            0.0178868, 0.05539798, 0.01374729, 0.0025297])

# Create a figure with subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Importance de la permutation des features pour chaque modèle', fontsize=16)

# Sort indices based on importances
adaboost_indices = np.argsort(importances_adaboost)[::-1]
gb_indices = np.argsort(importances_gb)[::-1]
rf_indices = np.argsort(importances_rf)[::-1]
svm_indices = np.argsort(importances_svm)[::-1]

# Plot for AdaBoost
axs[0, 0].bar(np.array(feature_names)[adaboost_indices], importances_adaboost[adaboost_indices], color='darkorange')
axs[0, 0].set_title('AdaBoost')

# Plot for Gradient Boost
axs[0, 1].bar(np.array(feature_names)[gb_indices], importances_gb[gb_indices], color='mediumorchid')
axs[0, 1].set_title('Gradient Boost')

# Plot for Random Forest
axs[1, 0].bar(np.array(feature_names)[rf_indices], importances_rf[rf_indices], color='forestgreen')
axs[1, 0].set_title('Random Forest')

# Plot for SVM
axs[1, 1].bar(np.array(feature_names)[svm_indices], importances_svm[svm_indices], color='skyblue')
axs[1, 1].set_title('SVM')

# Display the plot
plt.savefig('plots/pdf/permutations.pdf')