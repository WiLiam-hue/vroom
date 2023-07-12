import pickle
import streamlit as st
import numpy as np
import pandas as pd
import math
from PIL import Image

df = pickle.load(open('df.pkl', 'rb'))

# Access Euler's number
euler_number = math.e

# Define the available pickle files
pickle_files = {
    'Random Forest Regressor': 'trained_rfr1.pkl',
    'Extra Trees Regressor': 'trained_etr1.pkl',
    'K-Nearest Neighbour Regressor': 'trained_knn1.pkl',
    'Voting Regressor': 'trained_voting.pkl'
}

# Load the selected pickle file
selected_model = st.selectbox(
    'Select a Prediction Model', list(pickle_files.keys()))

model_path = pickle_files[selected_model]
pipe = pickle.load(open(model_path, 'rb'))

# Set the title based on the selected model
if selected_model == 'Random Forest Regressor':
    st.title("Random Forest Regressor :deciduous_tree:")
elif selected_model == 'Extra Trees Regressor':
    st.title("Extra Trees Regressor :evergreen_tree:")
elif selected_model == 'K-Nearest Neighbour Regressor':
    st.title("K-Nearest Neighbour Regressor :sparkle:")
elif selected_model == 'Voting Regressor':
    st.title("Voting Regressor :100:")

# brand
brand = st.selectbox('Vehicle Brand', df['brand'].unique())

# Filter models based on selected brand
filtered_models = df[df['brand'] == brand]['model'].unique()
model = st.selectbox('Vehicle Model', filtered_models)

# vehicleType
filtered_vehicle = df[df['model'] == model]['vehicleType'].unique()
vehicleType = st.selectbox('Vehicle Type', filtered_vehicle)

# gearbox
gearbox = st.selectbox('Gearbox Type', df['gearbox'].unique())

# fuelType
fuelType = st.selectbox('Fuel Type', df['fuelType'].unique())

# Get the minimum and maximum values of the "kilometer" feature from the DataFrame
kilometer_min = int(round(df['kilometer'].min()))
kilometer_max = int(round(df['kilometer'].max()))

# Create a slider for the "kilometer" feature
kilometer = st.slider("Select Kilometer", min_value=kilometer_min,
                      max_value=kilometer_max, step=1000)

# notRepairedDamage
notRepairedDamage = st.selectbox(
    'Repaired from Damage', df['notRepairedDamage'].unique())

# carAge
# Get the minimum and maximum values of the "carAge" feature from the DataFrame
carage_min = int(round(df['carAge'].min()))
carage_max = int(round(df['carAge'].max()))

# Create a slider for the "carAge" feature
carAge = st.slider("Select Car Age", min_value=carage_min,
                   max_value=carage_max, step=1)

# # powerPS_log
# Perform log transformation on 'powerPS' column
constant = df['powerPS_log'] * np.log10(math.e)
df['powerPS1'] = 10 ** constant

# Define the slider for powerPS
min_power = int(df['powerPS1'].min())
max_power = int(df['powerPS1'].max())

powerPS = st.slider("Select Car PowerPS", min_value=min_power,
                    max_value=max_power, step=1)

# transform powerPS to powerPS_log using log
powerPS_log = np.log(powerPS)

if selected_model == 'Random Forest Regressor':
    # Evaluation metrics for Random Forest Regressor
    test_set_r2 = 0.8632563489269417
    test_set_mae = 915.4609377877221
    test_set_mse = 2133407.327450648
    test_set_rmse = 1460.6188166152892
    image = Image.open('rf_fi.png')

elif selected_model == 'Extra Trees Regressor':
    # Evaluation metrics for Extra Trees Regressor
    test_set_r2 = 0.8529860033284948
    test_set_mae = 921.4428074937155
    test_set_mse = 2293640.2185811526
    test_set_rmse = 1514.4768795135674
    image = Image.open('etr_fi.png')
 
elif selected_model == 'K-Nearest Neighbour Regressor':
    # Evaluation metrics for K-Nearest Neighbors (KNN)
    test_set_r2 = 0.8369414490927543
    test_set_mae = 1001.7166595721199
    test_set_mse = 2543959.4787705746
    test_set_rmse = 1594.9794602973966
    image = Image.open('knn_fi.png')

elif selected_model == "Voting Regressor":
    test_set_r2 = 0.8641298212583445
    test_set_mae = 918.2732558181965
    test_set_mse = 2119779.840854254
    test_set_rmse = 1455.94637293214
    image = Image.open('voting_fi.png')

if st.button("Predict Price"):

    data = pd.DataFrame({
        'vehicleType': [vehicleType],
        'gearbox': [gearbox],
        'model': [model],
        'kilometer': [kilometer],
        'fuelType': [fuelType],
        'brand': [brand],
        'notRepairedDamage': [notRepairedDamage],
        'powerPS_log': [powerPS_log],
        'carAge': [carAge]
    })
    # Make predictions
    predictions = pipe.predict(data)
    # Format the predicted price
    predicted_price = "The predicted price is: € " + \
        str(round(predictions[0], 2))
    # Display the predicted price as the title
    st.title(predicted_price)

    # Display the evaluation metrics based on the selected model
    if selected_model == 'Random Forest Regressor':
        st.write("Test Set:")
        st.write(f"R2 score: {test_set_r2}")
        st.write(f"MAE: {test_set_mae}")
        st.write(f"MSE: {test_set_mse}")
        st.write(f"RMSE: {test_set_rmse}")
        st.image(image, caption='Feature Importance in Random Forest')

    elif selected_model == 'Extra Trees Regressor':
        st.write("Test Set:")
        st.write(f"R2 score: {test_set_r2}")
        st.write(f"MAE: {test_set_mae}")
        st.write(f"MSE: {test_set_mse}")
        st.write(f"RMSE: {test_set_rmse}")
        st.image(image, caption='Feature Importance in Extra Trees Regressor')

    elif selected_model == 'K-Nearest Neighbour Regressor':
        st.write("Test Set:")
        st.write(f"R2 score: {test_set_r2}")
        st.write(f"MAE: {test_set_mae}")
        st.write(f"MSE: {test_set_mse}")
        st.write(f"RMSE: {test_set_rmse}")
        st.image(image, caption='Feature Importance in KNN Regressor')
    
    elif selected_model == 'Voting Regressor':
        st.write("Test Set:")
        st.write(f"R2 score: {test_set_r2}")
        st.write(f"MAE: {test_set_mae}")
        st.write(f"MSE: {test_set_mse}")
        st.write(f"RMSE: {test_set_rmse}")
        st.image(image, caption='Feature Importance in Voting Regressor')
