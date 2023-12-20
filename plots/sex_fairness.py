import matplotlib.pyplot as plt

# Models
models = ['AdaBoost', 'Gradient Boosting', 'Random Forest', 'SVM']
colours = ['darkorange', 'mediumorchid', 'forestgreen', 'skyblue']

# Train Demographic Parity Difference
train_dem_parity = [0.1459, 0.1271, 0.1147, 0.1328]

# Train Equalised Odds Difference
train_eq_odds = [0.0832, 0.0599, 0.0025, 0.0842]

# Test Demographic Parity Difference
test_dem_parity = [0.1341, 0.1155, 0.1204, 0.1302]

# Test Equalised Odds Difference
test_eq_odds = [0.1198, 0.0920, 0.0934, 0.1201]

# Create subplots for train
fig, axs_train = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Différences d\'équité pour chaque modèle (Train)', fontsize=16)

# Train Parity
axs_train[0].bar(models, train_dem_parity, color=colours, alpha=0.7)
axs_train[0].set_title('Train demographic parity difference')
axs_train[0].set_ylabel('Différence')

# Train Equalised Odds
axs_train[1].bar(models, train_eq_odds, color=colours, alpha=0.7)
axs_train[1].set_title('Train equalised odds difference')
axs_train[1].set_ylabel('Différence')

# Adjust layout
plt.tight_layout()

# Show the train plot
plt.savefig('plots/pdf/train_sex_metrics.pdf')

# Create subplots for test
fig, axs_test = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Différences d\'équité pour chaque modèle (Test)', fontsize=16)

# Test Parity
axs_test[0].bar(models, test_dem_parity, color=colours, alpha=0.7)
axs_test[0].set_title('Test demographic parity difference')
axs_test[0].set_ylabel('Différence')

# Test Equalised Odds
axs_test[1].bar(models, test_eq_odds, color=colours, alpha=0.7)
axs_test[1].set_title('Test equalised odds difference')
axs_test[1].set_ylabel('Différence')

# Adjust layout
plt.tight_layout()

# Show the test plot
plt.savefig('plots/pdf/test_sex_metrics.pdf')



################ Same thing on the no SEX data ################
# Train Demographic Parity Difference
models = ['AdaBoost', 'Gradient Boosting', 'Random Forest', 'SVM']
train_dem_parity_no_sex = [0.0694, 0.0579, 0.1121, 0.0213]

# Train Equalised Odds Difference
train_eq_odds_no_sex = [0.0126, 0.0236, 0.0020, 0.0494]

# Test Demographic Parity Difference
test_dem_parity_no_sex = [0.0624, 0.0643, 0.0494, 0.0304]

# Test Equalised Odds Difference
test_eq_odds_no_sex = [0.0351, 0.0292, 0.0200, 0.0386]

# Create subplots for train
fig, axs_train = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Différences d\'équité pour chaque modèle (Train sans la feature SEX)', fontsize=16)

# Train Parity
axs_train[0].bar(models, train_dem_parity_no_sex, color=colours, alpha=0.7)
axs_train[0].set_title('Train demographic parity difference')
axs_train[0].set_ylabel('Différence')

# Train Equalised Odds
axs_train[1].bar(models, train_eq_odds_no_sex, color=colours, alpha=0.7)
axs_train[1].set_title('Train equalised odds difference')
axs_train[1].set_ylabel('Différence')

# Adjust layout
plt.tight_layout()

# Show the train plot
plt.savefig('plots/pdf/train_sex_metrics_no_sex.pdf')

# Create subplots for test
fig, axs_test = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Différences d\'équité pour chaque modèle (Test sans la feature SEX)', fontsize=16)

# Test Parity
axs_test[0].bar(models, test_dem_parity_no_sex, color=colours, alpha=0.7)
axs_test[0].set_title('Test demographic parity difference')
axs_test[0].set_ylabel('Différence')

# Test Equalised Odds
axs_test[1].bar(models, test_eq_odds_no_sex, color=colours, alpha=0.7)
axs_test[1].set_title('Test equalised odds difference')
axs_test[1].set_ylabel('Différence')

# Adjust layout
plt.tight_layout()

# Show the test plot
plt.savefig('plots/pdf/test_sex_metrics_no_sex.pdf')