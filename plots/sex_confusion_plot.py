import matplotlib.pyplot as plt
import numpy as np

models = ['AdaBoost', 'Gradient Boost', 'Random Forest', 'SVM']
bar_width = 0.35
bar_positions_women = np.arange(len(models))
bar_positions_men = bar_positions_women + bar_width

# Données d'entraînement
ada_train_f = np.array([[8399, 1154], [1574, 3648]])
ada_train_m = np.array([[7054, 1719], [1693, 6065]])

gb_train_f = np.array([[8343, 1210], [1428, 3794]])
gb_train_m = np.array([[7175, 1598], [1657, 6101]])

rf_train_f = np.array([[9471, 82], [63, 5159]])
rf_train_m = np.array([[8697, 76], [74, 7684]])

svm_train_f = np.array([[8183, 1370], [1779, 3443]])
svm_train_m = np.array([[6961, 1812], [1990, 5768]])

# Données de test
ada_test_f = np.array([[2085, 318], [425, 856]])
ada_test_m = np.array([[1874, 415], [393, 1461]])

gb_test_f = np.array([[2072, 331], [379, 902]])
gb_test_m = np.array([[1900, 389], [378, 1476]])

rf_test_f = np.array([[2083, 320], [396, 885]])
rf_test_m = np.array([[1889, 400], [400, 1454]])

svm_test_f = np.array([[2050, 353], [435, 846]])
svm_test_m = np.array([[1848, 441], [407, 1447]])

# Précision pour femmes et hommes
train_precision_women = [ada_train_f[1, 1] / np.sum(ada_train_f[:, 1]),
                         gb_train_f[1, 1] / np.sum(gb_train_f[:, 1]),
                         rf_train_f[1, 1] / np.sum(rf_train_f[:, 1]),
                         svm_train_f[1, 1] / np.sum(svm_train_f[:, 1])]

train_precision_men = [ada_train_m[1, 1] / np.sum(ada_train_m[:, 1]),
                      gb_train_m[1, 1] / np.sum(gb_train_m[:, 1]),
                      rf_train_m[1, 1] / np.sum(rf_train_m[:, 1]),
                      svm_train_m[1, 1] / np.sum(svm_train_m[:, 1])]

test_precision_women = [ada_test_f[1, 1] / np.sum(ada_test_f[:, 1]),
                        gb_test_f[1, 1] / np.sum(gb_test_f[:, 1]),
                        rf_test_f[1, 1] / np.sum(rf_test_f[:, 1]),
                        svm_test_f[1, 1] / np.sum(svm_test_f[:, 1])]

test_precision_men = [ada_test_m[1, 1] / np.sum(ada_test_m[:, 1]),
                      gb_test_m[1, 1] / np.sum(gb_test_m[:, 1]),
                      rf_test_m[1, 1] / np.sum(rf_test_m[:, 1]),
                      svm_test_m[1, 1] / np.sum(svm_test_m[:, 1])]

# Rappel pour femmes et hommes
train_recall_women = [ada_train_f[1, 1] / np.sum(ada_train_f[1, :]),
                      gb_train_f[1, 1] / np.sum(gb_train_f[1, :]),
                      rf_train_f[1, 1] / np.sum(rf_train_f[1, :]),
                      svm_train_f[1, 1] / np.sum(svm_train_f[1, :])]

train_recall_men = [ada_train_m[1, 1] / np.sum(ada_train_m[1, :]),
                    gb_train_m[1, 1] / np.sum(gb_train_m[1, :]),
                    rf_train_m[1, 1] / np.sum(rf_train_m[1, :]),
                    svm_train_m[1, 1] / np.sum(svm_train_m[1, :])]

test_recall_women = [ada_test_f[1, 1] / np.sum(ada_test_f[1, :]),
                     gb_test_f[1, 1] / np.sum(gb_test_f[1, :]),
                     rf_test_f[1, 1] / np.sum(rf_test_f[1, :]),
                     svm_test_f[1, 1] / np.sum(svm_test_f[1, :])]

test_recall_men = [ada_test_m[1, 1] / np.sum(ada_test_m[1, :]),
                   gb_test_m[1, 1] / np.sum(gb_test_m[1, :]),
                   rf_test_m[1, 1] / np.sum(rf_test_m[1, :]),
                   svm_test_m[1, 1] / np.sum(svm_test_m[1, :])]

print("Train differences M/F:")
[print("{:.1f}".format((train_precision_men[i] - train_precision_women[i]) * 100)) for i in range(4)]
print('\n')
[print("{:.1f}".format((train_recall_men[i] - train_recall_women[i]) * 100)) for i in range(4)]

print("\nTest differences M/F:")
[print("{:.1f}".format((test_precision_men[i] - test_precision_women[i]) * 100)) for i in range(4)]
print('\n')
[print("{:.1f}".format((test_recall_men[i] - test_recall_women[i]) * 100)) for i in range(4)]

for i, (precision_women, precision_men, recall_women, recall_men) in enumerate([(train_precision_women, train_precision_men, train_recall_women, train_recall_men), (test_precision_women, test_precision_men, test_recall_women, test_recall_men)]):
    # Création du graphique
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.bar(bar_positions_women, precision_women, width=bar_width, label='Women', color='mediumpurple')
    ax1.bar(bar_positions_men, precision_men, width=bar_width, label='Men', color='mediumseagreen')
    ax1.set_ylabel('Précision')
    ax1.set_title('Précision des modèles pour les femmes et les hommes')
    ax1.legend()

    # Graphique du rappel
    ax2.bar(bar_positions_women, recall_women, width=bar_width, label='Women', color='mediumpurple')
    ax2.bar(bar_positions_men, recall_men, width=bar_width, label='Men', color='mediumseagreen')
    ax2.set_ylabel('Rappel')
    ax2.set_title('Rappel des modèles pour les femmes et les hommes')
    ax2.legend()

    # Ajout ticks et grille
    for ax in [ax1, ax2]:
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=0.7)

    data = "d'entraînement"
    data_title = "train"
    if i == 1: 
        data = "de test"
        data_title = "test"
    plt.suptitle("Mesures sur les données " + data)
    # Afficher le graphique
    plt.savefig(f'plots/pdf/sex_confusion_{data_title}.pdf')







################ Same thing on the no SEX data ################

# Données d'entraînement
ada_train_f = np.array([[8100, 1453], [1302, 3920]])
ada_train_m = np.array([[7340, 1433], [2032, 5726]])

gb_train_f = np.array([[8070, 1483], [1215, 4007]])
gb_train_m = np.array([[7443, 1330], [1988, 5770]])

rf_train_f = np.array([[9436, 117], [72, 5150]])
rf_train_m = np.array([[8683, 90], [102, 7656]])

svm_train_f = np.array([[7649, 1904], [1346, 3876]])
svm_train_m = np.array([[7329, 1444], [2383, 5375]])

# Données de test
ada_test_f = np.array([[2012, 391], [369, 912]])
ada_test_m = np.array([[1950, 339], [469, 1385]])

gb_test_f = np.array([[2015, 388], [331, 950]])
gb_test_m = np.array([[1947, 342], [452, 1429]])

rf_test_f = np.array([[2020, 383], [338, 943]])
rf_test_m = np.array([[1970, 319], [477, 1377]])

svm_test_f = np.array([[1948, 455], [347, 934]])
svm_test_m = np.array([[1944, 345], [511, 1343]])

# Précision pour femmes et hommes
train_precision_women = [ada_train_f[1, 1] / np.sum(ada_train_f[:, 1]),
                         gb_train_f[1, 1] / np.sum(gb_train_f[:, 1]),
                         rf_train_f[1, 1] / np.sum(rf_train_f[:, 1]),
                         svm_train_f[1, 1] / np.sum(svm_train_f[:, 1])]

train_precision_men = [ada_train_m[1, 1] / np.sum(ada_train_m[:, 1]),
                      gb_train_m[1, 1] / np.sum(gb_train_m[:, 1]),
                      rf_train_m[1, 1] / np.sum(rf_train_m[:, 1]),
                      svm_train_m[1, 1] / np.sum(svm_train_m[:, 1])]

test_precision_women = [ada_test_f[1, 1] / np.sum(ada_test_f[:, 1]),
                        gb_test_f[1, 1] / np.sum(gb_test_f[:, 1]),
                        rf_test_f[1, 1] / np.sum(rf_test_f[:, 1]),
                        svm_test_f[1, 1] / np.sum(svm_test_f[:, 1])]

test_precision_men = [ada_test_m[1, 1] / np.sum(ada_test_m[:, 1]),
                      gb_test_m[1, 1] / np.sum(gb_test_m[:, 1]),
                      rf_test_m[1, 1] / np.sum(rf_test_m[:, 1]),
                      svm_test_m[1, 1] / np.sum(svm_test_m[:, 1])]

# Rappel pour femmes et hommes
train_recall_women = [ada_train_f[1, 1] / np.sum(ada_train_f[1, :]),
                      gb_train_f[1, 1] / np.sum(gb_train_f[1, :]),
                      rf_train_f[1, 1] / np.sum(rf_train_f[1, :]),
                      svm_train_f[1, 1] / np.sum(svm_train_f[1, :])]

train_recall_men = [ada_train_m[1, 1] / np.sum(ada_train_m[1, :]),
                    gb_train_m[1, 1] / np.sum(gb_train_m[1, :]),
                    rf_train_m[1, 1] / np.sum(rf_train_m[1, :]),
                    svm_train_m[1, 1] / np.sum(svm_train_m[1, :])]

test_recall_women = [ada_test_f[1, 1] / np.sum(ada_test_f[1, :]),
                     gb_test_f[1, 1] / np.sum(gb_test_f[1, :]),
                     rf_test_f[1, 1] / np.sum(rf_test_f[1, :]),
                     svm_test_f[1, 1] / np.sum(svm_test_f[1, :])]

test_recall_men = [ada_test_m[1, 1] / np.sum(ada_test_m[1, :]),
                   gb_test_m[1, 1] / np.sum(gb_test_m[1, :]),
                   rf_test_m[1, 1] / np.sum(rf_test_m[1, :]),
                   svm_test_m[1, 1] / np.sum(svm_test_m[1, :])]

print("\nTrain differences M/F - no sex:")
[print("{:.1f}".format((train_precision_men[i] - train_precision_women[i]) * 100)) for i in range(4)]
print('\n')
[print("{:.1f}".format((train_recall_men[i] - train_recall_women[i]) * 100)) for i in range(4)]

print("\nTest differences M/F - no sex:")
[print("{:.1f}".format((test_precision_men[i] - test_precision_women[i]) * 100)) for i in range(4)]
print('\n')
[print("{:.1f}".format((test_recall_men[i] - test_recall_women[i]) * 100)) for i in range(4)]

for i, (precision_women, precision_men, recall_women, recall_men) in enumerate([(train_precision_women, train_precision_men, train_recall_women, train_recall_men), (test_precision_women, test_precision_men, test_recall_women, test_recall_men)]):
    # Création du graphique
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.bar(bar_positions_women, precision_women, width=bar_width, label='Women', color='mediumpurple')
    ax1.bar(bar_positions_men, precision_men, width=bar_width, label='Men', color='mediumseagreen')
    ax1.set_ylabel('Précision')
    ax1.set_title('Précision des modèles pour les femmes et les hommes')
    ax1.legend()

    # Graphique du rappel
    ax2.bar(bar_positions_women, recall_women, width=bar_width, label='Women', color='mediumpurple')
    ax2.bar(bar_positions_men, recall_men, width=bar_width, label='Men', color='mediumseagreen')
    ax2.set_ylabel('Rappel')
    ax2.set_title('Rappel des modèles pour les femmes et les hommes')
    ax2.legend()

    # Ajout ticks et grille
    for ax in [ax1, ax2]:
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=0.7)

    data = "d'entraînement"
    data_title = "train"
    if i == 1: 
        data = "de test"
        data_title = "test"
    plt.suptitle("Mesures sur les données " + data + " sans la feature SEX")
    # Afficher le graphique
    plt.savefig(f'plots/pdf/sex_confusion_{data_title}_no_sex.pdf')