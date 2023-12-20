import matplotlib.pyplot as plt
import numpy as np

models = ['Ada', 'GB', 'RF', 'SVM']
bar_width = 0.35
bar_positions_women = np.arange(len(models))
bar_positions_men = bar_positions_women + bar_width

def get_metrics(data):
    precision_women = [data[model][0][1, 1] / np.sum(data[model][0][:, 1]) for model in models]
    recall_women = [data[model][0][1, 1] / np.sum(data[model][0][1, :]) for model in models]
    precision_men = [data[model][1][1, 1] / np.sum(data[model][1][:, 1]) for model in models]
    recall_men = [data[model][1][1, 1] / np.sum(data[model][1][1, :]) for model in models]
    return precision_women, recall_women, precision_men, recall_men

def plot_metrics(data, split, sexFeature):
    for precision_women, recall_women, precision_men, recall_men in data:
        # Création du graphique
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.bar(bar_positions_women, precision_women, width=bar_width, label='Women', color='mediumpurple')
        ax1.bar(bar_positions_men, precision_men, width=bar_width, label='Men', color='mediumseagreen')
        ax1.set_ylabel('Précision')
        ax1.set_title('Précision des modèles sur SEX')
        ax1.legend()

        # Graphique du rappel
        ax2.bar(bar_positions_women, recall_women, width=bar_width, label='Women', color='mediumpurple')
        ax2.bar(bar_positions_men, recall_men, width=bar_width, label='Men', color='mediumseagreen')
        ax2.set_ylabel('Rappel')
        ax2.set_title('Rappel des modèles sur SEX')
        ax2.legend()

        # Ajout ticks et grille
        for ax in [ax1, ax2]:
            # Center the ticks between the two bars for each model
            tick_positions = bar_positions_women + bar_width / 2
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(models)
            ax.set_yticks(np.arange(0, 1.1, 0.1))
            ax.set_axisbelow(True)
            ax.grid(axis='y', alpha=0.7)

        plt.suptitle(f"Mesures sur les données de {split}{' sans la feature SEX' if not sexFeature else ''}")
        # Afficher le graphique
        plt.savefig(f"plots/pdf/sex_confusion_{split}{'_no_sex' if not sexFeature else ''}.pdf", bbox_inches='tight')

# Original data (F matrix, M matrix)
train_data = {
    'Ada': [np.array([[8399, 1154], [1574, 3648]]), np.array([[7054, 1719], [1693, 6065]])],
    'GB': [np.array([[8343, 1210], [1428, 3794]]), np.array([[7175, 1598], [1657, 6101]])],
    'RF': [np.array([[9471, 82], [63, 5159]]), np.array([[8697, 76], [74, 7684]])],
    'SVM': [np.array([[8183, 1370], [1779, 3443]]), np.array([[6961, 1812], [1990, 5768]])]
}

test_data = {
    'Ada': [np.array([[2085, 318], [425, 856]]), np.array([[1874, 415], [393, 1461]])],
    'GB': [np.array([[2072, 331], [379, 902]]), np.array([[1900, 389], [378, 1476]])],
    'RF': [np.array([[2083, 320], [396, 885]]), np.array([[1889, 400], [400, 1454]])],
    'SVM': [np.array([[2050, 353], [435, 846]]), np.array([[1848, 441], [407, 1447]])]
}

train_data_no_sex = {
    'Ada': [np.array([[8100, 1453], [1302, 3920]]), np.array([[7340, 1433], [2032, 5726]])],
    'GB': [np.array([[8070, 1483], [1215, 4007]]), np.array([[7443, 1330], [1988, 5770]])],
    'RF': [np.array([[9436, 117], [72, 5150]]), np.array([[8683, 90], [102, 7656]])],
    'SVM': [np.array([[7649, 1904], [1346, 3876]]), np.array([[7329, 1444], [2383, 5375]])]
}

test_data_no_sex = {
    'Ada': [np.array([[2012, 391], [369, 912]]), np.array([[1950, 339], [469, 1385]])],
    'GB': [np.array([[2015, 388], [331, 950]]), np.array([[1947, 342], [452, 1429]])],
    'RF': [np.array([[2020, 383], [338, 943]]), np.array([[1970, 319], [477, 1377]])],
    'SVM': [np.array([[1948, 455], [347, 934]]), np.array([[1944, 345], [511, 1343]])]
}

data_sets = [
    (train_data, 'train', True),
    (test_data, 'test', True),
    (train_data_no_sex, 'train', False),
    (test_data_no_sex, 'test', False)
]

for data, split, sexFeature in data_sets:
    metrics_data = [get_metrics(data)]
    plot_metrics(metrics_data, split, sexFeature)
