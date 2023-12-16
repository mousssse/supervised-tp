import sys
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_selection import r_regression
from sklearn.inspection import permutation_importance
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

class Logger:
    """To write both to the file and the terminal"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

    def close(self):
        self.log.close()

# Define the output file
output_file = 'tp_final_results.txt'

# Redirect sys.stdout to the Logger
sys.stdout = Logger(output_file)

# To retrieve categories after sandardisation
sex_mapping = {1: "Male", 2: "Female"}
rac1p_mapping = {1: "White", 2: "Black / African American", 3: "American Indian", 4: "Alaska Native", 5: "Am. Indian and/or Alaska Native", 6: "Asian", 7: "Native Hawaiian", 8: "Other race alone", 9: "Two+ races"}

def preprocess():
    # Load the dataset
    features = pd.read_csv('acsincome_ca_features.csv')
    labels = pd.read_csv('acsincome_ca_labels.csv')

    # Concatenate in one dataset
    data = pd.concat([features, labels], axis=1)

    # View dataset
    # print(data.head())

    data_no_sex = data.drop('SEX', axis=1)
    data_no_rac1p = data.drop('RAC1P', axis=1)

    sample = 0.2
    print(f'Analysis with a sample of size {sample*100}% -> {int(len(data) * sample)} examples\n\n')

    splits = []
    for i, d in enumerate([data, data_no_sex, data_no_rac1p]):
        # Create a random sample of the dataset
        data_sample = d.sample(frac=sample, random_state=42)

        # Divide features and labels back
        x = data_sample.drop('PINCP', axis=1)
        y = data_sample['PINCP']

        # Standardise the data
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        # Additional steps on the full dataset
        if i == 0:
            sex_id = x.columns.get_loc('SEX') if 'SEX' in x.columns else None
            rac1p_id = x.columns.get_loc('RAC1P') if 'RAC1P' in x.columns else None

            # Create dictionaries to map standardized values to original values for SEX and RAC1P
            for value, original_value in zip(x_scaled[:, sex_id], x['SEX']):
                sex_mapping[int(original_value)] = value
            for value, original_value in zip(x_scaled[:, rac1p_id], x['RAC1P']):   
                rac1p_mapping[value] = int(original_value)

        # Split train/test
        size = 0.2
        k = int(1/size)
        splits.append(train_test_split(x_scaled, y, test_size=size, random_state=42))
    
    return splits, k, sex_id, rac1p_id
    

def evaluate_model(model, grid_search, x_train, x_test, y_train, y_test, k, sex_id, rac1p_id):
    # Cross validation
    cv_scores = cross_val_score(model, x_train, y_train, cv=k)

    # Print scores :)
    print(f"Cross validation scores: {cv_scores}. Mean: {cv_scores.mean()}")

    # Train the model and measure the runtime
    start = time.time()
    model.fit(x_train, y_train)
    end = time.time()
    print(f"Training time: {end - start} seconds")

    # Test the model on test and train data
    y_pred = model.predict(x_test)
    y_train_res = model.predict(x_train)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    classif_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Accuracy: {accuracy}\nClassification report:\n{classif_report}\nConfusion matrix:\n{conf_matrix}")

    # Grid search
    grid_start = time.time()
    grid_search.fit(x_train, y_train)
    grid_end = time.time()
    print(f"Grid search time: {grid_end - grid_start} seconds")

    # Best params
    print(f"Best parameters for {model}:", grid_search.best_params_)

    best_model = grid_search.best_estimator_
    y_pred_best = best_model.predict(x_test)

    # Metrics
    accuracy_best = accuracy_score(y_test, y_pred_best)
    classif_report_best = classification_report(y_test, y_pred_best)
    conf_matrix_best = confusion_matrix(y_test, y_pred_best)
    print(f"Accuracy: {accuracy_best}\nClassification report:\n{classif_report_best}\nConfusion matrix:\n{conf_matrix_best}")

    # Partie 2.1
    correlations_pred = r_regression(x_test, y_pred_best)
    print("Prediction correlation\n", correlations_pred)

    perm_imp = permutation_importance(best_model, x_test, y_test, random_state=42)
    print("Permutation importance\n", perm_imp)

    # Partie 2.2 
    if sex_id is not None:
        print("----------------------sex feature metrics----------------------")
        # Comparing to the sex_mapping values, 1 being male and 2 being female
        f_matrix = confusion_matrix(y_test[x_test[:, sex_id] == sex_mapping[2]], y_pred_best[x_test[:, sex_id] == sex_mapping[2]])
        m_matrix = confusion_matrix(y_test[x_test[:, sex_id] == sex_mapping[1]], y_pred_best[x_test[:, sex_id] == sex_mapping[1]])

        print("Confusion matrix F\n", f_matrix)
        print("Confusion matrix M\n", m_matrix)
        
        # fairness metrics
        dp_diff_train = demographic_parity_difference(y_train, y_train_res, sensitive_features=x_train[:, sex_id])
        eod_diff_train = equalized_odds_difference(y_train, y_train_res, sensitive_features=x_train[:, sex_id])

        dp_diff_test = demographic_parity_difference(y_test, y_pred_best, sensitive_features=x_test[:, sex_id])
        eod_diff_test = equalized_odds_difference(y_test, y_pred_best, sensitive_features=x_test[:, sex_id])

        print("Train demographic parity difference:", dp_diff_train)
        print("Test demographic parity difference:", dp_diff_test)

        print("Train equalised odds difference:", eod_diff_train)
        print("Test equalised odds difference:", eod_diff_test)

    if rac1p_id is not None:
        print("----------------------race feature metrics----------------------")
        rac1p_categories = sorted(np.unique(x_test[:, rac1p_id]))

        for category in rac1p_categories:
            # Retrieving corresponding name
            category_name = rac1p_mapping[rac1p_mapping[category]]

            # Confusion matrix for each rac1p category
            category_indices = x_test[:, rac1p_id] == category
            category_conf_matrix = confusion_matrix(y_test[category_indices], y_pred_best[category_indices])
            print(f"Confusion matrix for RAC1P category {category_name}:\n{category_conf_matrix}")

            # Fairness metrics for each rac1p category
            category_indices_train = x_train[:, rac1p_id] == category
            category_indices_test = x_test[:, rac1p_id] == category

            dp_diff_train = demographic_parity_difference(y_train, y_train_res, sensitive_features=category_indices_train)
            eod_diff_train = equalized_odds_difference(y_train, y_train_res, sensitive_features=category_indices_train)

            dp_diff_test = demographic_parity_difference(y_test, y_pred_best, sensitive_features=category_indices_test)
            eod_diff_test = equalized_odds_difference(y_test, y_pred_best, sensitive_features=category_indices_test)

            print(f"Train demographic parity difference for RAC1P category {category_name}: {dp_diff_train}")
            print(f"Test demographic parity difference for RAC1P category {category_name}: {dp_diff_test}")

            print(f"Train equalised odds difference for RAC1P category {category_name}: {eod_diff_train}")
            print(f"Test equalised odds difference for RAC1P category {category_name}: {eod_diff_test}\n")


splits, k, sex_id, rac1p_id = preprocess()
[[x_train, x_test, y_train, y_test], [x_train_no_sex, x_test_no_sex, y_train_no_sex, y_test_no_sex], [x_train_no_rac1p, x_test_no_rac1p, y_train_no_rac1p, y_test_no_rac1p]] = splits

correlations = r_regression(x_train, y_train)
print("Correlations :\n", correlations)


####################### Models evaluation #######################
models = []

param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5]
}
grid_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=k)
models.append(("Random Forest", RandomForestClassifier(), grid_rf))

param_grid_ada = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.1, 1]
}
grid_ada = GridSearchCV(AdaBoostClassifier(), param_grid_ada, cv=k)
models.append(("Ada Boost", AdaBoostClassifier(), grid_ada))

param_grid_gb = {
    'n_estimators': [50, 100], 
    'learning_rate': [0.01, 0.1, 1],
    'max_depth': [3, 5]
}
grid_gb = GridSearchCV(GradientBoostingClassifier(), param_grid_gb, cv=k)
models.append(("Gradient", GradientBoostingClassifier(), grid_gb))

param_grid_svm = {
    'C': [0.1, 1], #params régularisation
    'kernel': ['linear', 'rbf'], # type de noyau
    'gamma': ['scale', 'auto']
}
grid_svm = GridSearchCV(SVC(), param_grid_svm, cv=k)
models.append(("SVM", SVC(kernel='linear'), grid_svm))

for (model_name, model, grid_search) in models:
    print(f"\n--------------------------{model_name}---------------------------")
    evaluate_model(model, grid_search, x_train, x_test, y_train, y_test, k, sex_id, rac1p_id)
    print(f"\n--------------------{model_name} without sex---------------------")
    evaluate_model(model, grid_search, x_train_no_sex, x_test_no_sex, y_train_no_sex, y_test_no_sex, k, None, None)
    print(f"\n-------------------{model_name} without rac1p--------------------")
    evaluate_model(model, grid_search, x_train_no_rac1p, x_test_no_rac1p, y_train_no_rac1p, y_test_no_rac1p, k, None, None)
    print("---------------------------------------------------\n\n\n")