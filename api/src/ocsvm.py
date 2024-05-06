import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.speciescodes import speciescode_to_name

scaler = StandardScaler()

oc_svm = joblib.load('models/one_class_svm_model.pkl')

testing_data = np.load('models/testing_data.npz')
X_test_loaded = testing_data['X_test']
column_transformer = ColumnTransformer(
    [('encoder', OneHotEncoder(handle_unknown='ignore'), ['SPECIES_CODE'])],
    remainder='passthrough'
)

def distance_to_decision_boundary(decision_scores):
    distances = np.abs(decision_scores)

    if np.min(distances) == np.max(distances):
        normalized_distances = np.full_like(distances, 0.5)
    else:
        normalized_distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))

    prob_scores = 1 - normalized_distances
    return prob_scores


def get_species_code(english_name, speciescode_to_name):
    for species_code, american_english_name in speciescode_to_name:
        if american_english_name == english_name:
            return species_code

def get_probability(latitude, longitude, month, speciescode):
    speciescode = get_species_code(speciescode, speciescode_to_name)
    speciescode = 'daejun'

    print(latitude, longitude, month, speciescode)

    new_data = [[latitude, longitude, month, speciescode]]

    new_data_point = pd.DataFrame(new_data, columns=['LATITUDE', 'LONGITUDE', 'Month', 'SPECIES_CODE'])

    X_test_loaded_df = pd.DataFrame(X_test_loaded, columns=['LATITUDE', 'LONGITUDE', 'Month', 'SPECIES_CODE'])

    new_data_point_encoded = column_transformer.fit_transform(new_data_point)

    new_data_scaled = scaler.fit_transform(new_data_point_encoded)

    new_data_scaled = np.array(new_data_scaled).reshape(1, -1)

    X_test_with_new_data = np.vstack([X_test_loaded_df, new_data_scaled])

    decision_scores_combined = oc_svm.decision_function(X_test_with_new_data)

    probability_new_data = distance_to_decision_boundary(decision_scores_combined)[-1]

    prediction = oc_svm.predict(new_data_scaled)

    print("Prediction (1 for sighting, -1 for non-sighting):", prediction, probability_new_data)

    return prediction[0], probability_new_data
