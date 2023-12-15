import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_selection import r_regression
from sklearn.inspection import permutation_importance
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

def preprocess():
    # Load the dataset
    features = pd.read_csv('acsincome_ca_features.csv')
    labels = pd.read_csv('acsincome_ca_labels.csv')

    # Concatenate in one dataset
    data = pd.concat([features, labels], axis=1)

    # View dataset
    # print(data.head())

    # Create a random sample of the dataset
    sample = 0.01
    data_sample = data.sample(frac=sample, random_state=42)

    # Divide features and labels back
    x = data_sample.drop('PINCP', axis=1)
    y = data_sample['PINCP']

    sex_id = x.columns.get_loc('SEX')

    # Standardise the data
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Split train/test
    size = 0.2
    k = int(1/size)
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=size, random_state=42)
    return x_train, x_test, y_train, y_test, k, sex_id
    


def evaluate_model(model, grid_search, x_train, x_test, y_train, y_test, k, sex_id):
    # Cross validation
    cv_scores = cross_val_score(model, x_train, y_train, cv=k)

    # Print scores :)
    print(f"Cross validation scores: {cv_scores}. Mean: {cv_scores.mean()}")

    # Evaluate quality
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_train_res = model.predict(x_train)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    classif_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Accuracy: {accuracy}\nClassification report:\n{classif_report}\nConfusion matrix:\n{conf_matrix}")

    grid_search.fit(x_train, y_train)

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
    # Comparing to 0 because the values have been standardised
    f_matrix = confusion_matrix(y_test[x_test[:, sex_id] < 0], y_pred_best[x_test[:, sex_id] < 0])
    m_matrix = confusion_matrix(y_test[x_test[:, sex_id] > 0], y_pred_best[x_test[:, sex_id] > 0])

    print("Confusion matrix F\n", f_matrix)
    print("Confusion matrix M\n", m_matrix)
    
    dp_diff_train = demographic_parity_difference(y_train, y_train_res, sensitive_features=x_train[:, sex_id])
    eod_diff_train = equalized_odds_difference(y_train, y_train_res, sensitive_features=x_train[:, sex_id])

    dp_diff_test = demographic_parity_difference(y_test, y_pred_best, sensitive_features=x_test[:, sex_id])
    eod_diff_test = equalized_odds_difference(y_test, y_pred_best, sensitive_features=x_test[:, sex_id])

    print("Train demographic parity difference:", dp_diff_train)
    print("Test demographic parity difference:", dp_diff_test)

    print("Train equalised odds difference:", eod_diff_train)
    print("Test equalised odds difference:", eod_diff_test)


x_train, x_test, y_train, y_test, k, sex_id = preprocess()

correlations = r_regression(x_train, y_train)
print("Correlations :\n", correlations)

param_grid_svm = {
    'C': [0.1, 1], #params régularisation
    'kernel': ['linear', 'rbf'], # type de noyau
    'gamma': ['scale', 'auto']
}
grid_svm = GridSearchCV(SVC(), param_grid_svm, cv=k)

param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5]
}
grid_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=k)

param_grid_ada = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.1, 1]
}
grid_ada = GridSearchCV(AdaBoostClassifier(), param_grid_ada, cv=k)

param_grid_gb = {
    'n_estimators': [50, 100], 
    'learning_rate': [0.01, 0.1, 1],
    'max_depth': [3, 5]
}
grid_gb = GridSearchCV(GradientBoostingClassifier(), param_grid_gb, cv=k)

print("---------------------SVM-----------------------")
evaluate_model(SVC(kernel='linear'), grid_svm, x_train, x_test, y_train, y_test, k, sex_id)
print("---------------------------------------------------\n")

# print("-----------------Random Forest-----------------")
# evaluate_model(RandomForestClassifier(), grid_rf, x_train, x_test, y_train, y_test, k, sex_id)
# print("---------------------------------------------------\n")

# print("---------------------Ada-----------------------")
# evaluate_model(AdaBoostClassifier(), grid_ada, x_train, x_test, y_train, y_test, k, sex_id)
# print("---------------------------------------------------\n")

# print("-------------------Gradient--------------------")
# evaluate_model(GradientBoostingClassifier(), grid_gb, x_train, x_test, y_train, y_test, k, sex_id)
# print("---------------------------------------------------\n")
