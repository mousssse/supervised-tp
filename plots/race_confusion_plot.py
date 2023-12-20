import matplotlib.pyplot as plt
import numpy as np

models = ['Ada Boost', 'Gradient Boosting', 'Random Forest', 'SVM']
rac1p_categories = [
    'White',
    'Black / African American',
    'American Indian',
    'Am. Indian and/or Alaska Native',
    'Asian',
    'Native Hawaiian',
    'Other race alone',
    'Two+ races'
] # Note: ignoring Alaska Natives values bc only 1 example

bar_width = 0.1
bar_positions = np.arange(len(models))

def get_metrics_rac1p(data):
    precision_categories = {model: [] for model in models}
    recall_categories = {model: [] for model in models}

    for model in models:
        for i in range(len(rac1p_categories)):
            true_positive = data[model][i][1, 1]
            false_positive = data[model][i][0, 1]
            false_negative = data[model][i][1, 0]

            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

            precision_categories[model].append(precision)
            recall_categories[model].append(recall)

    return precision_categories, recall_categories

def plot_metrics(data, rac1pFeature):
    precision_by_model, recall_by_model = data

    # Create the figure with subplots
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot precision
    for i, category in enumerate(rac1p_categories):
        precision_values = [precision_by_model[model][i] for model in models]
        ax1.bar(bar_positions + i * bar_width, np.array(precision_values), width=bar_width, label=category)

    ax1.set_ylabel('Précision')
    ax1.set_title('Précision des modèles sur RAC1P')
    ax1.set_xticks(bar_positions + bar_width * (len(rac1p_categories) - 1) / 2)
    ax1.set_xticklabels(models)

    # Plot recall
    for i, category in enumerate(rac1p_categories):
        recall_values = [recall_by_model[model][i] for model in models]
        ax2.bar(bar_positions + i * bar_width, np.array(recall_values), width=bar_width, label=category)

    ax2.set_ylabel('Rappel')
    ax2.set_title('Rappel des modèles sur RAC1P')
    ax2.set_xticks(bar_positions + bar_width * (len(rac1p_categories) - 1) / 2)
    ax2.set_xticklabels(models)
    lgd = ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Move legend to the right

    # Add grid and save the figure
    for ax in [ax1, ax2]:
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=0.7)

    plt.suptitle(f"Métriques sur les données de test{' sans la feature RAC1P' if not rac1pFeature else ''}")
    plt.tight_layout()
    plt.savefig(f"plots/pdf/rac1p_confusion_test{'_no_rac1p' if not rac1pFeature else ''}.pdf", bbox_extra_artists=(lgd,))

# Test data for RAC1P
test_data_rac1p = {
    'Ada Boost': [
        np.array([[2286, 464], [489, 1610]]),
        np.array([[194, 57], [30, 77]]),
        np.array([[40, 3], [3, 11]]),
        np.array([[12, 1], [2, 2]]),
        np.array([[578, 144], [138, 462]]),
        np.array([[17, 5], [0, 5]]),
        np.array([[675, 38], [111, 68]]),
        np.array([[157, 21], [45, 81]])
    ],
    'Gradient Boosting': [
        np.array([[2297, 453], [455, 1644]]),
        np.array([[200, 51], [31, 76]]),
        np.array([[41, 2], [3, 11]]),
        np.array([[11, 2], [2, 2]]),
        np.array([[579, 143], [115, 485]]),
        np.array([[17, 5], [0, 5]]),
        np.array([[670, 43], [109, 70]]),
        np.array([[157, 21], [42, 84]])
    ],
    'Random Forest': [
        np.array([[2291, 459], [479, 1620]]),
        np.array([[198, 53], [35, 72]]),
        np.array([[40, 3], [3, 11]]),
        np.array([[12, 1], [2, 2]]),
        np.array([[581, 141], [124, 476]]),
        np.array([[20, 2], [1, 4]]),
        np.array([[673, 40], [109, 70]]),
        np.array([[157, 21], [43, 83]])
    ],
    'SVM': [
        np.array([[2248, 502], [505, 1594]]),
        np.array([[195, 56], [41, 66]]),
        np.array([[38, 5], [3, 11]]),
        np.array([[12, 1], [2, 2]]),
        np.array([[562, 160], [131, 469]]),
        np.array([[18, 4], [1, 4]]),
        np.array([[670, 43], [113, 66]]),
        np.array([[155, 23], [46, 80]])
    ]
}

# Test data without RAC1P feature
test_data_no_rac1p = {
    'Ada Boost': [
        np.array([[2312, 438], [514, 1585]]),
        np.array([[198, 53], [32, 75]]),
        np.array([[40, 3], [3, 11]]),
        np.array([[12, 1], [2, 2]]),
        np.array([[562, 160], [127, 473]]),
        np.array([[15, 7], [0, 5]]),
        np.array([[668, 45], [107, 72]]),
        np.array([[152, 26], [40, 86]])
    ],
    'Gradient Boosting': [
        np.array([[2305, 445], [453, 1646]]),
        np.array([[198, 53], [31, 76]]),
        np.array([[41, 2], [3, 11]]),
        np.array([[12, 1], [2, 2]]),
        np.array([[576, 146], [116, 484]]),
        np.array([[17, 5], [0, 5]]),
        np.array([[658, 55], [99, 80]]),
        np.array([[153, 25], [38, 88]])
    ],
    'Random Forest': [
        np.array([[2317, 433], [493, 1606]]),
        np.array([[197, 54], [36, 71]]),
        np.array([[39, 4], [3, 11]]),
        np.array([[12, 1], [2, 2]]),
        np.array([[578, 144], [126, 474]]),
        np.array([[19, 3], [1, 4]]),
        np.array([[659, 54], [103, 76]]),
        np.array([[154, 24], [40, 86]])
    ],
    'SVM': [
        np.array([[2263, 487], [526, 1573]]),
        np.array([[198, 53], [41, 66]]),
        np.array([[37, 6], [3, 11]]),
        np.array([[12, 1], [2, 2]]),
        np.array([[561, 161], [138, 462]]),
        np.array([[18, 4], [0, 5]]),
        np.array([[663, 50], [103, 76]]),
        np.array([[150, 28], [40, 86]])
    ]
}

data_sets_rac1p = [
    (test_data_rac1p, True),
    (test_data_no_rac1p, False)
]

for data, rac1pFeature in data_sets_rac1p:
    metrics_data_rac1p = get_metrics_rac1p(data)
    plot_metrics(metrics_data_rac1p, rac1pFeature)
