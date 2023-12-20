import matplotlib.pyplot as plt
import numpy as np

models = ['Ada', 'GB', 'RF', 'SVM']
rac1p_categories = [
    'White',
    'Black / African American',
    'American Indian',
    'Am. Indian and/or Alaska Native',
    'Asian',
    'Native Hawaiian',
    'Other race alone',
    'Two+ races'
]  # Note: ignoring Alaska Natives values bc only 1 example

bar_width = 0.1
bar_positions = np.arange(len(models))

def plot_rates(data, rac1pFeature):
    tpr_by_model = {model: [] for model in models}
    tnr_by_model = {model: [] for model in models}

    for model, rates in data.items():
        for i, category in enumerate(rac1p_categories):
            tpr_by_model[model].append(rates[i]['TPR'])
            tnr_by_model[model].append(rates[i]['TNR'])

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Plot TPR
    for i, category in enumerate(rac1p_categories):
        tpr_values = [model[i] for model in tpr_by_model.values()]
        ax1.bar(bar_positions + i * bar_width, np.array(tpr_values), width=bar_width, label=category)

    ax1.set_ylabel('Taux de vrais positifs (TPR)')
    ax1.set_title('TPR des modèles sur RAC1P')
    ax1.set_xticks(bar_positions + bar_width * (len(rac1p_categories) - 1) / 2)
    ax1.set_xticklabels(models)

    # Plot TNR
    for i, category in enumerate(rac1p_categories):
        tnr_values = [model[i] for model in tnr_by_model.values()]
        ax2.bar(bar_positions + i * bar_width, np.array(tnr_values), width=bar_width, label=category)

    ax2.set_ylabel('Taux de vrais négatifs (TNR)')
    ax2.set_title('TNR des modèles sur RAC1P')
    ax2.set_xticks(bar_positions + bar_width * (len(rac1p_categories) - 1) / 2)
    ax2.set_xticklabels(models)
    lgd = ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Move legend to the right

    for ax in [ax1, ax2]:
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=0.7)

    plt.suptitle(f"Taux vrais positifs/négatifs en test{' sans la feature RAC1P' if not rac1pFeature else ''}")
    plt.tight_layout()
    plt.savefig(f"plots/pdf/rac1p_rates_test{'_no_rac1p' if not rac1pFeature else ''}.pdf", bbox_extra_artists=(lgd,))

# Test rates for RAC1P
test_rates_rac1p = {
    'Ada': [
        {'TPR': 0.7670, 'TNR': 0.8313}, {'TPR': 0.7196, 'TNR': 0.7729},
        {'TPR': 0.7857, 'TNR': 0.9302}, {'TPR': 0.5000, 'TNR': 0.9231},
        {'TPR': 0.7700, 'TNR': 0.8006}, {'TPR': 1.0000, 'TNR': 0.7727},
        {'TPR': 0.3799, 'TNR': 0.9467}, {'TPR': 0.6429, 'TNR': 0.8820}
    ],
    'GB': [
        {'TPR': 0.7832, 'TNR': 0.8353}, {'TPR': 0.7103, 'TNR': 0.7968},
        {'TPR': 0.7857, 'TNR': 0.9535}, {'TPR': 0.5000, 'TNR': 0.8462},
        {'TPR': 0.8083, 'TNR': 0.8019}, {'TPR': 1.0000, 'TNR': 0.7727},
        {'TPR': 0.3911, 'TNR': 0.9397}, {'TPR': 0.6667, 'TNR': 0.8820}
    ],
    'RF': [
        {'TPR': 0.7718, 'TNR': 0.8331}, {'TPR': 0.6729, 'TNR': 0.7888},
        {'TPR': 0.7857, 'TNR': 0.9302}, {'TPR': 0.5000, 'TNR': 0.9231},
        {'TPR': 0.7933, 'TNR': 0.8047}, {'TPR': 0.8000, 'TNR': 0.9091},
        {'TPR': 0.3911, 'TNR': 0.9439}, {'TPR': 0.6587, 'TNR': 0.8820}
    ],
    'SVM': [
        {'TPR': 0.7594, 'TNR': 0.8175}, {'TPR': 0.6168, 'TNR': 0.7769},
        {'TPR': 0.7857, 'TNR': 0.8837}, {'TPR': 0.5000, 'TNR': 0.9231},
        {'TPR': 0.7817, 'TNR': 0.7784}, {'TPR': 0.8000, 'TNR': 0.8182},
        {'TPR': 0.3687, 'TNR': 0.9397}, {'TPR': 0.6349, 'TNR': 0.8708}
    ]
}

# Test rates without RAC1P
test_rates_no_rac1p = {
    'Ada': [
        {'TPR': 0.7551, 'TNR': 0.8407}, {'TPR': 0.7009, 'TNR': 0.7888},
        {'TPR': 0.7857, 'TNR': 0.9302}, {'TPR': 0.5000, 'TNR': 0.9231},
        {'TPR': 0.7883, 'TNR': 0.7784}, {'TPR': 1.0000, 'TNR': 0.6818},
        {'TPR': 0.4022, 'TNR': 0.9369}, {'TPR': 0.6825, 'TNR': 0.8539}
    ],
    'GB': [
        {'TPR': 0.7842, 'TNR': 0.8382}, {'TPR': 0.7103, 'TNR': 0.7888},
        {'TPR': 0.7857, 'TNR': 0.9535}, {'TPR': 0.5000, 'TNR': 0.9231},
        {'TPR': 0.8067, 'TNR': 0.7978}, {'TPR': 1.0000, 'TNR': 0.7727},
        {'TPR': 0.4469, 'TNR': 0.9229}, {'TPR': 0.6984, 'TNR': 0.8596}
    ],
    'RF': [
        {'TPR': 0.7651, 'TNR': 0.8425}, {'TPR': 0.6636, 'TNR': 0.7849},
        {'TPR': 0.7857, 'TNR': 0.9070}, {'TPR': 0.5000, 'TNR': 0.9231},
        {'TPR': 0.7900, 'TNR': 0.8006}, {'TPR': 0.8000, 'TNR': 0.8636},
        {'TPR': 0.4246, 'TNR': 0.9243}, {'TPR': 0.6825, 'TNR': 0.8652}
    ],
    'SVM': [
        {'TPR': 0.7494, 'TNR': 0.8229}, {'TPR': 0.6168, 'TNR': 0.7888},
        {'TPR': 0.7857, 'TNR': 0.8605}, {'TPR': 0.5000, 'TNR': 0.9231},
        {'TPR': 0.7700, 'TNR': 0.7770}, {'TPR': 1.0000, 'TNR': 0.8182},
        {'TPR': 0.4246, 'TNR': 0.9299}, {'TPR': 0.6825, 'TNR': 0.8427}
    ]
}

# For each data set (with and without RAC1P feature), plot the rates
data_sets_rac1p = [
    (test_rates_rac1p, True),
    (test_rates_no_rac1p, False)
]

for data, rac1pFeature in data_sets_rac1p:
    plot_rates(data, rac1pFeature)
