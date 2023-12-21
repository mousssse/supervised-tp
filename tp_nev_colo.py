import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

# Charger les données de Californie
california_features = pd.read_csv('acsincome_ca_features.csv')
california_labels = pd.read_csv('acsincome_ca_labels.csv')

# Concaténer les données en un seul ensemble de données
california_data = pd.concat([california_features, california_labels], axis=1)

# Diviser les données en fonctionnalités et étiquettes
x_california = california_data.drop('PINCP', axis=1)
y_california = california_data['PINCP']

# Standardiser les données
scaler = StandardScaler()
x_california_scaled = scaler.fit_transform(x_california)

# Diviser les données en ensembles d'entraînement et de test
x_california_train, x_california_test, y_california_train, y_california_test = train_test_split(
    x_california_scaled, y_california, test_size=0.2, random_state=42
)

# Entraîner un modèle (par exemple, Random Forest)
model = RandomForestClassifier()
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(x_california_train, y_california_train)

# Afficher les métriques pour le dataset californien
print("Métriques pour le dataset californien :")
y_california_pred = grid_search.predict(x_california_test)
print(f"Accuracy : {accuracy_score(y_california_test, y_california_pred):.4f}")
print(classification_report(y_california_test, y_california_pred))
print(confusion_matrix(y_california_test, y_california_pred))

# Charger les données du Nevada et du Colorado
nevada_features = pd.read_csv('complementary_data/acsincome_ne_allfeaturesTP2.csv')
nevada_labels = pd.read_csv('complementary_data/acsincome_ne_labelTP2.csv')
colorado_features = pd.read_csv('complementary_data/acsincome_co_allfeaturesTP2.csv')
colorado_labels = pd.read_csv('complementary_data/acsincome_co_labelTP2.csv')

# Concaténer les données du Nevada en un seul ensemble de données
nevada_data = pd.concat([nevada_features, nevada_labels], axis=1)

# Concaténer les données du Colorado en un seul ensemble de données
colorado_data = pd.concat([colorado_features, colorado_labels], axis=1)

# Diviser les données en fonctionnalités et étiquettes pour le Nevada et le Colorado
x_nevada = scaler.transform(nevada_data.drop('PINCP', axis=1))
y_nevada = nevada_data['PINCP']

x_colorado = scaler.transform(colorado_data.drop('PINCP', axis=1))
y_colorado = colorado_data['PINCP']

# Faire des prédictions sur le Nevada et afficher les métriques
y_nevada_pred = grid_search.predict(x_nevada)
print("\nMétriques pour les prédictions sur le Nevada :")
print(f"Accuracy : {accuracy_score(y_nevada, y_nevada_pred):.4f}")
print(classification_report(y_nevada, y_nevada_pred))
print(confusion_matrix(y_nevada, y_nevada_pred))

# Faire des prédictions sur le Colorado et afficher les métriques
y_colorado_pred = grid_search.predict(x_colorado)
print("\nMétriques pour les prédictions sur le Colorado :")
print(f"Accuracy : {accuracy_score(y_colorado, y_colorado_pred):.4f}")
print(classification_report(y_colorado, y_colorado_pred))
print(confusion_matrix(y_colorado, y_colorado_pred))
