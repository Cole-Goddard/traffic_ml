# App to predict traffic volume using pre-trained ML models in Streamlit

# Import libraries
import streamlit as st
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

# Create title
st.title('Traffic Volume Prediction: A Machine Learning App')

# Display the gif
st.image('traffic_image.gif', width = 700)

# Create first subheader
st.subheader("Utilize our advanced Machine Learning application to predict traffic volume.")

# Create text for user guidance
st.write('Use the following form to get started')

# Asking users to input their data using a form
# We can use st.form() and st.submit_form_button() to wrap the rest of user inputs in and allow the user to change all of the inputs and submit the entire form at once instead of multiple times
# Adding Streamlit functions to get user input
with st.form('user_inputs'):
    # Loading traffic_df
    traffic_df = pd.read_csv('Traffic_Volume.csv')
    # Drop weather_description column because it is not needed
    traffic_df = traffic_df.drop('weather_description', axis=1)

    # Replace nan values with 'None' in holiday column
    traffic_df['holiday'].fillna('None', inplace=True)
    # selectbox for holiday (categorical variable)
    # (NOTE: Make sure that variable names are same as that of training dataset)
    holiday = st.selectbox('Choose whether today is a designated holiday or not', options=traffic_df['holiday'].unique())
    
    # number_input for temp, rain_1h, snow_1h, 
    # numerical variables (NOTE: Make sure that variable names are same as that of training dataset)
    temp = st.number_input('Average temperature in Kelvin', min_value=0.00)
    rain_1h = st.number_input('Amount in mm of rain that occurred in the hour', min_value=0.00)
    snow_1h = st.number_input('Amount in mm of snow that occurred in the hour', min_value=0.00)
    clouds_all = st.number_input('Percentage of cloud cover', min_value=0, max_value = 100)

    # selectbox for weather_main (categorical variable)
    # (NOTE: Make sure that variable names are same as that of training dataset)
    weather_main = st.selectbox('Choose the current weather', options=traffic_df['weather_main'].unique())

    # Convert the 'date_time' column to datetime format
    traffic_df['date_time'] = pd.to_datetime(traffic_df['date_time'])
    # Using date_time column, create new columns for month, weekday, and hour
    traffic_df['month'] = traffic_df['date_time'].dt.strftime('%B')
    traffic_df['weekday'] = traffic_df['date_time'].dt.strftime('%A')
    traffic_df['hour'] = traffic_df['date_time'].dt.hour
    # Order selectbox options
    ordered_months = [month.capitalize() for month in pd.date_range(start='2022-01-01', end='2022-12-01', freq='M').strftime('%B')]
    ordered_weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    ordered_hours = list(range(24))
    # selectbox for month, weekday, hour (categorical variables)
    # (NOTE: Make sure that variable names are same as that of training dataset)
    month = st.selectbox('Choose the month', options=ordered_months)
    weekday = st.selectbox('Choose the day of the week', options=ordered_weekdays)
    hour = st.selectbox('Choose the hour', options=ordered_hours)

    # selectbox for model (categorical variable)
    model = st.selectbox('Select Machine Learning Model for Prediction', options=['Decision Tree', 'Random Forest', 'AdaBoost', 'XGBoost'])

    # Create text to label predictive DataFrame
    st.write("These ML models exhibited the following predictive performance on the test dataset.")
    # Create data for DataFrame (hard coded from traffic.ipynb)
    predictive_data = {
        'ML Model': ['Decision Tree', 'Random Forest', 'AdaBoost', 'XGBoost'],
        'R2': [0.939472, 0.940816, 0.945778, 0.945941],
        'RMSE': [469.092756, 469.048632, 440.815439, 441.452954]
    }
    # Create DataFrame listing R2 and RMSE values for each algorithm on test dataset
    predictive_df = pd.DataFrame(predictive_data)
    # Define a function to apply conditional styling
    def color_cells(val):
        if val == 'Decision Tree' or val == 'Random Forest' or val == 0.939472 or val == 0.940816 or val == 469.092756 or val == 469.048632:
            return 'background-color: orange'
        elif val == 'AdaBoost' or val == 'XGBoost' or val == 0.945778 or val == 0.945941 or val == 440.815439 or val == 441.452954:
            return 'background-color: lime'
        else:
            return ''
    # Apply the styling function to the columns
    styled_predictive_df = predictive_df.style.applymap(color_cells)
    # Display the DataFrame in Streamlit, with index hidden and ustom formatting for float columns
    #st.dataframe(styled_predictive_df.style.format({'R2': '{:.6f}', 'RMSE': '{:.6f}'}), hide_index=True)
    #st.dataframe(styled_predictive_df.style.format({'R2': '{:.6f}', 'RMSE': '{:.6f}'}), hide_index=True)
    st.dataframe(styled_predictive_df, hide_index=True)

    # Create submit button
    st.form_submit_button() 

# Create list of user inputs from form
user_inputs = [holiday, temp, rain_1h, snow_1h, clouds_all, weather_main, month, weekday, hour]

# Loading original_df
original_df = pd.read_csv('Traffic_Volume.csv')
# Remove weather_description column because it is not needed
original_df = original_df.drop('weather_description', axis=1)
# Remove output (traffic_volume) column from original_df
original_df = original_df.drop(columns = ['traffic_volume'])
# Convert the 'date_time' column to datetime format
original_df['date_time'] = pd.to_datetime(original_df['date_time'])
# Using date_time column, create new columns for month, weekday, and hour
original_df['month'] = original_df['date_time'].dt.strftime('%B')
original_df['weekday'] = original_df['date_time'].dt.strftime('%A')
original_df['hour'] = original_df['date_time'].dt.hour
# Remove date_time column from original_df
original_df = original_df.drop(columns = ['date_time'])

# Create copy of original_df for use in concatenation
combined_df = original_df.copy()
# Concatenate user_input with original_df
combined_df.loc[len(combined_df)] = user_inputs

# Number of rows in combined_df
#combined_rows = combined_df.shape[0]

# Create dummies for the combined dataframe
categorical_variables = ['holiday', 'weather_main', 'month', 'weekday', 'hour']
combined_df_encoded = pd.get_dummies(combined_df, columns=categorical_variables)

# create df of just encoded user inputs
user_inputs_df_encoded = combined_df_encoded.tail(1)

# Load the correct models based on the selected model in the form
if model == 'Decision Tree':
    # Reading the dt_pickle file that was created in traffic.ipynb 
    dt_pickle = open('decision_tree.pickle', 'rb') 
    reg = pickle.load(dt_pickle) 
    dt_pickle.close()
    # Text for Decision Tree
    text = "Decision Tree Traffic Prediction: "
    # set image_name
    image_name = 'dt_feature_importance.svg'
elif model == 'Random Forest':
    # Reading the rf_pickle file that was created in traffic.ipynb
    rf_pickle = open('random_forest.pickle', 'rb')
    reg = pickle.load(rf_pickle) 
    rf_pickle.close() 
    # Text for Random Forest
    text = "Random Forest Traffic Prediction: "
    # set image_name
    image_name = 'rf_feature_importance.svg'
elif model == 'AdaBoost':
    # Reading the ab_pickle file that was created in traffic.ipynb
    ab_pickle = open('adaboost.pickle', 'rb') 
    reg = pickle.load(ab_pickle) 
    ab_pickle.close()
    # Text for AdaBoost
    text = "AdaBoost Traffic Prediction: "
    # set image_name
    image_name = 'ab_feature_importance.svg'
elif model == 'XGBoost':
    # Reading the xgb_pickle file that was created in traffic.ipynb
    xgb_pickle = open('xgboost.pickle', 'rb') 
    reg = pickle.load(xgb_pickle) 
    xgb_pickle.close() 
    # Text for XGBoost
    text = "XGBoost Traffic Prediction: "
    # set image_name
    image_name = 'xgb_feature_importance.svg'

# Using predict() with new data provided by the user in the form
user_pred = reg.predict(user_inputs_df_encoded)
# Show the predicted traffic volume on the app
st.write(text + str(int(user_pred)), color="red")

# Create second subheader
st.subheader("Plot of Feature Performance")
# Showing Feature Importance Plot
st.image(image_name)