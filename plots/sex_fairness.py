import matplotlib.pyplot as plt

def plot_metrics(models, metrics, title, filename):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=16)

    # Plot Demographic Parity Difference
    axs[0].bar(models, metrics[0], color=colours, alpha=0.7)
    axs[0].set_title('Différence de parité démographique')
    axs[0].set_ylabel('Différence')

    # Plot Equalised Odds Difference
    axs[1].bar(models, metrics[1], color=colours, alpha=0.7)
    axs[1].set_title('Différence d\'égalité des chances')
    axs[1].set_ylabel('Différence')

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(filename)

# Models
models = ['AdaBoost', 'Gradient Boosting', 'Random Forest', 'SVM']
colours = ['darkorange', 'mediumorchid', 'forestgreen', 'skyblue']

# Train metrics
train_dem_parity = [0.1459, 0.1271, 0.1147, 0.1328]
train_eq_odds = [0.0832, 0.0599, 0.0025, 0.0842]

# Test metrics
test_dem_parity = [0.1341, 0.1155, 0.1204, 0.1302]
test_eq_odds = [0.1198, 0.0920, 0.0934, 0.1201]

# Plot train metrics with SEX
plot_metrics(models, [train_dem_parity, train_eq_odds], 'Différences d\'équité pour chaque modèle (Train)', 'plots/pdf/train_sex_metrics.pdf')

# Plot test metrics with SEX
plot_metrics(models, [test_dem_parity, test_eq_odds], 'Différences d\'équité pour chaque modèle (Test)', 'plots/pdf/test_sex_metrics.pdf')



# Train metrics without SEX
train_dem_parity_no_sex = [0.0694, 0.0579, 0.1121, 0.0213]
train_eq_odds_no_sex = [0.0126, 0.0236, 0.0020, 0.0494]

# Test metrics without SEX
test_dem_parity_no_sex = [0.0624, 0.0643, 0.0494, 0.0304]
test_eq_odds_no_sex = [0.0351, 0.0292, 0.0200, 0.0386]

# Plot train metrics without SEX
plot_metrics(models, [train_dem_parity_no_sex, train_eq_odds_no_sex], 'Différences d\'équité pour chaque modèle (Train sans la feature SEX)', 'plots/pdf/train_sex_metrics_no_sex.pdf')

# Plot test metrics without SEX
plot_metrics(models, [test_dem_parity_no_sex, test_eq_odds_no_sex], 'Différences d\'équité pour chaque modèle (Test sans la feature SEX)', 'plots/pdf/test_sex_metrics_no_sex.pdf')