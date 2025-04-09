import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def load_data(file_path):
    data = []
    labels = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()            
            values = line.split(',')
            try:
                values = list(map(float, values))
            except ValueError:
                print(f" Erreur de conversion dans la ligne : {line}")
                continue           
            data.append(values[:-3])
            labels.append(tuple(map(int, values[-3:])))
    return np.array(data), np.array(labels)

file_path = "all_coef.txt"
X, y = load_data(file_path)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


mlp = MLPClassifier(hidden_layer_sizes=(12, 8, 6), activation='relu', 
                    max_iter=1000, tol=1e-4, random_state=42)

mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle (3 couches optimisées) : {accuracy:.2f}")

joblib.dump(mlp, "mlp_model_3couches.pkl")
print("Modèle enregistré sous 'mlp_model_3couches.pkl'")

