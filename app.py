import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from joblib import load
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

# create sidebar
with st.sidebar:
    st.title("Car Evaluation using KNN Model")
    st.markdown('''
    ### About 
    Lorem ipsum dolor sit amet.
                
    ### Libraries
    - [Streamlit](https://streamlit.io/)
    - [scikit-learn](https://scikit-learn.org/stable/index.html)
    - [Joblib](https://joblib.readthedocs.io/en/stable/)
    - [Matplotlib](), [Seaborn](), [numpy](), [pandas]()
                
    ### How it works:
    1. Takes data about your car
    2. Encodes into processable data
    3. Use the model to predict evaluation
    4. Present the result to the user
    ''')

    add_vertical_space(3)
    st.write("Author: Muhammad Naufal Al Ghifari")

def encode_data(data, category_order):
    categories = list(category_order.values())
    encoder = OrdinalEncoder(categories=categories)
    encoded_data = encoder.fit_transform(data)
    return encoded_data

def predict_car(model, encoded_data):
    prediction = model.predict(encoded_data)
    return prediction

def main():
    # Load the trained model
    model_path = './Models/rf_car_eval.joblib'
    knn_model = load(model_path)

    # Define the order of categorical data in ascending order
    categorical_orders = {
        'buying':['low', 'med', 'high', 'vhigh'],
        'maintenance':['low', 'med', 'high', 'vhigh'],
        'doors':['2', '3', '4', '5more'],
        'persons':['2', '4', 'more'],
        'trunk':['small', 'med', 'big'],
        'safety':['low', 'med', 'high'],
    }

    # define the classes
    classes = {0:'Unacceptable', 1:'Acceptable', 2:'Good', 3:'Very Good'}

    st.title("Evaluate your car using a KNN Classification model 🚗 ✅")
    st.write(f"**Loaded model**: {model_path}")

    # Create input fields for each feature
    buying = st.selectbox('Buying price', options=categorical_orders['buying'])
    maintenance = st.selectbox('Maintenance cost', options=categorical_orders['maintenance'])
    doors = st.selectbox('Number of Doors', options=categorical_orders['doors'])
    persons = st.selectbox('Number of Persons', options=categorical_orders['persons'])
    trunk = st.selectbox('Trunk size', options=categorical_orders['trunk'])
    safety = st.selectbox('Safety level', options=categorical_orders['safety'])

    # Make a prediction
    if st.button('Calculate Prediction'):
        # set input data
        input_data = np.array([[buying, maintenance, doors, persons, trunk, safety]])

        # encode data
        enc_data = encode_data(input_data, categorical_orders)

        # predict data
        prediction = predict_car(knn_model, enc_data)

        st.markdown("""---""")
        st.markdown(f"**Your car's predicted Safety Level**: {classes[int(prediction[0])]}")

if __name__ == "__main__":
    main()