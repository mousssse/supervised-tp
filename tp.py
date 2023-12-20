import sys
import pandas as pd
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
rac1p_categories = {1: "White", 2: "Black / African American", 3: "American Indian", 4: "Alaska Native", 5: "Am. Indian and/or Alaska Native", 6: "Asian", 7: "Native Hawaiian", 8: "Other race alone", 9: "Two+ races"}

def preprocess():
    # Load the dataset
    features = pd.read_csv('acsincome_ca_features.csv')
    labels = pd.read_csv('acsincome_ca_labels.csv')

    # Concatenate in one dataset
    data = pd.concat([features, labels], axis=1)

    # View dataset
    # print(data.head())

    sample = 0.2
    # Create a random sample of the dataset
    data_sample = data.sample(frac=sample, random_state=42)
    print(f'Analysis with a sample of size {sample*100}% -> {int(len(data) * sample)} examples\n\n')

    # Divide features and labels back
    x = data_sample.drop('PINCP', axis=1)
    y = data_sample['PINCP']

    # Split train/test
    size = 0.2
    k = int(1/size)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=size, random_state=42)

    # Saving sex and race indices
    global sex_mapping_train, sex_mapping_test, rac1p_mapping_train, rac1p_mapping_test
    sex_mapping_train = x_train['SEX'].map({1: 'Male', 2: 'Female'}).reset_index(drop=True)
    sex_mapping_test = x_test['SEX'].map({1: 'Male', 2: 'Female'}).reset_index(drop=True)
    rac1p_mapping_train = x_train['RAC1P'].map(rac1p_categories).reset_index(drop=True)
    rac1p_mapping_test = x_test['RAC1P'].map(rac1p_categories).reset_index(drop=True)

    splits = []
    for i, (x_train_val, x_test_val) in enumerate([(x_train, x_test), (x_train.drop('SEX', axis=1), x_test.drop('SEX', axis=1)), (x_train.drop('RAC1P', axis=1), x_test.drop('RAC1P', axis=1))]):
        # Standardise the data
        scaler = StandardScaler()
        scaler.fit(x_train_val)
        x_train_scaled = scaler.transform(x_train_val)
        x_test_scaled = scaler.transform(x_test_val)

        # Removing the first column which corresponds to the dataset index
        splits.append([x_train_scaled[:, 1:], x_test_scaled[:, 1:], y_train, y_test])

        if i == 0:
            # Saving sex values of the original full data
            global sex_column_train, sex_column_test
            sex_column_train = x_train_scaled[:, x.columns.get_loc('SEX')]
            sex_column_test = x_test_scaled[:, x.columns.get_loc('SEX')]
    
    return splits, k


def fairlearn_metrics(y_train, y_train_res, sensitive_col_train, y_test, y_pred_best, sensitive_col_test, category):
    num_train = len(sensitive_col_train)
    num_test = len(sensitive_col_test)

    # fairness metrics
    if num_train > 1:
        dp_diff_train = demographic_parity_difference(y_train, y_train_res, sensitive_features=sensitive_col_train)
        eod_diff_train = equalized_odds_difference(y_train, y_train_res, sensitive_features=sensitive_col_train)
        print(f"Train demographic parity difference for {category}: {dp_diff_train:.4f}")
        print(f"Train equalised odds difference for {category}: {eod_diff_train:.4f}")
    else: print("Not enough data to measure train metrics")
    
    if num_test > 1:
        dp_diff_test = demographic_parity_difference(y_test, y_pred_best, sensitive_features=sensitive_col_test)
        eod_diff_test = equalized_odds_difference(y_test, y_pred_best, sensitive_features=sensitive_col_test)
        print(f"Test demographic parity difference for {category}: {dp_diff_test:.4f}")
        print(f"Test equalised odds difference for {category}: {eod_diff_test:.4f}\n")
    else: print("Not enough data to measure test metrics\n")


def evaluate_model(model, grid_search, x_train, x_test, y_train, y_test, k, sexMetrics = True, rac1pMetrics = True):
    # Cross validation
    cv_scores = cross_val_score(model, x_train, y_train, cv=k)

    # Print scores :)
    print(f"Cross validation scores: [{', '.join([f'{n:.4f}' for n in cv_scores])}]. Mean: {cv_scores.mean():.4f}")

    # Train the model and measure the runtime
    start = time.time()
    model.fit(x_train, y_train)
    end = time.time()
    print(f"Training time: {end - start:.4f} seconds")

    # Test the model on test and train data
    y_pred = model.predict(x_test)
    y_train_res = model.predict(x_train)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    classif_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}\nClassification report:\n{classif_report}\nConfusion matrix:\n{conf_matrix}")

    # Grid search
    grid_start = time.time()
    grid_search.fit(x_train, y_train)
    grid_end = time.time()
    print(f"Grid search time: {grid_end - grid_start:.4f} seconds")

    # Best params
    print(f"Best parameters for {model}:", grid_search.best_params_)

    best_model = grid_search.best_estimator_
    y_pred_best = best_model.predict(x_test)

    # Metrics
    accuracy_best = accuracy_score(y_test, y_pred_best)
    classif_report_best = classification_report(y_test, y_pred_best)
    conf_matrix_best = confusion_matrix(y_test, y_pred_best)
    print(f"Accuracy: {accuracy_best:.4f}\nClassification report:\n{classif_report_best}\nConfusion matrix:\n{conf_matrix_best}")

    # Partie 2.1
    correlations_pred = r_regression(x_test, y_pred_best)
    print(f"Prediction correlations:\n[{', '.join([f'{n:.4f}' for n in correlations_pred])}]")

    perm_imp = permutation_importance(best_model, x_test, y_test, random_state=42)
    print("Permutation importance\n", perm_imp)

    # Partie 2.2 
    if sexMetrics:
        print("----------------------sex feature metrics----------------------")
        female_indices_train = sex_mapping_train[sex_mapping_train == 'Female'].index
        female_indices_test = sex_mapping_test[sex_mapping_test == 'Female'].index
        y_trainF = y_train.iloc[female_indices_train]
        y_train_resF = y_train_res[female_indices_train]
        y_testF = y_test.iloc[female_indices_test]
        y_pred_bestF = y_pred_best[female_indices_test]

        # Female matrices
        f_matrix_train = confusion_matrix(y_trainF, y_train_resF)
        f_matrix_test = confusion_matrix(y_testF, y_pred_bestF)

        print("Train Confusion matrix F\n", f_matrix_train)
        print("Test Confusion matrix F\n", f_matrix_test)
        
        male_indices_train = sex_mapping_train[sex_mapping_train == 'Male'].index
        male_indices_test = sex_mapping_test[sex_mapping_test == 'Male'].index
        y_trainM = y_train.iloc[male_indices_train]
        y_train_resM = y_train_res[male_indices_train]
        y_testM = y_test.iloc[male_indices_test]
        y_pred_bestM = y_pred_best[male_indices_test]

        # Male matrices
        m_matrix_train = confusion_matrix(y_trainM, y_train_resM)
        m_matrix_test = confusion_matrix(y_testM, y_pred_bestM)

        print("Train Confusion matrix M\n", m_matrix_train)
        print("Test Confusion matrix M\n", m_matrix_test)
        fairlearn_metrics(y_train, y_train_res, sex_column_train, y_test, y_pred_best, sex_column_test, "SEX")

    if rac1pMetrics:
        print("----------------------race feature metrics----------------------")
        for category in rac1p_categories.values():
            # Train/Test Confusion matrices for each race category
            category_indices_train = rac1p_mapping_train[rac1p_mapping_train == category].index
            category_indices_test = rac1p_mapping_test[rac1p_mapping_test == category].index
            y_trainCat = y_train.iloc[category_indices_train]
            y_train_resCat = y_train_res[category_indices_train]
            y_testCat = y_test.iloc[category_indices_test]
            y_pred_bestCat = y_pred_best[category_indices_test]

            category_conf_matrix_train = confusion_matrix(y_trainCat, y_train_resCat)
            print(f"Train Confusion matrix for RAC1P category {category}:\n{category_conf_matrix_train}")

            category_conf_matrix_test = confusion_matrix(y_testCat, y_pred_bestCat)
            print(f"Test Confusion matrix for RAC1P category {category}:\n{category_conf_matrix_test}")

            # Calculate True Positive/Negative Rates
            if len(category_conf_matrix_train) > 1:
                TPR_train = category_conf_matrix_train[1, 1] / (category_conf_matrix_train[1, 1] + category_conf_matrix_train[1, 0])
                TNR_train = category_conf_matrix_train[0, 0] / (category_conf_matrix_train[0, 0] + category_conf_matrix_train[0, 1])
                print(f"Train rates for {category} - TPR: {TPR_train:.4f}, TNR: {TNR_train:.4f}")
            else: print("Not enough data to measure train rates")
            if len(category_conf_matrix_test) > 1:
                TPR_test = category_conf_matrix_test[1, 1] / (category_conf_matrix_test[1, 1] + category_conf_matrix_test[1, 0])
                TNR_test = category_conf_matrix_test[0, 0] / (category_conf_matrix_test[0, 0] + category_conf_matrix_test[0, 1])
                print(f"Test rates for {category} - TPR: {TPR_test:.4f}, TNR: {TNR_test:.4f}\n")
            else: print("Not enough data to measure test rates\n")


splits, k = preprocess()
[[x_train, x_test, y_train, y_test], [x_train_no_sex, x_test_no_sex, y_train_no_sex, y_test_no_sex], [x_train_no_rac1p, x_test_no_rac1p, y_train_no_rac1p, y_test_no_rac1p]] = splits

correlations = r_regression(x_train, y_train)
print(f"Correlations:\n[{', '.join([f'{n:.4f}' for n in correlations])}]")


####################### Models evaluation #######################
models = []

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

param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5]
}
grid_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=k)
models.append(("Random Forest", RandomForestClassifier(), grid_rf))

param_grid_svm = {
    'C': [0.1, 1], #params régularisation
    'kernel': ['linear', 'rbf'], # type de noyau
    'gamma': ['scale', 'auto']
}
grid_svm = GridSearchCV(SVC(), param_grid_svm, cv=k)
models.append(("SVM", SVC(kernel='linear'), grid_svm))

for (model_name, model, grid_search) in models:
    print(f"\n--------------------------{model_name}---------------------------")
    evaluate_model(model, grid_search, x_train, x_test, y_train, y_test, k)
    print(f"\n--------------------{model_name} without sex---------------------")
    evaluate_model(model, grid_search, x_train_no_sex, x_test_no_sex, y_train_no_sex, y_test_no_sex, k, rac1pMetrics=False)
    print(f"\n-------------------{model_name} without rac1p--------------------")
    evaluate_model(model, grid_search, x_train_no_rac1p, x_test_no_rac1p, y_train_no_rac1p, y_test_no_rac1p, k, sexMetrics=False)
    print("---------------------------------------------------\n\n\n")