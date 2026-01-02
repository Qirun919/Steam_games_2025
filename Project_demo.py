import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
le = LabelEncoder()

st.set_page_config('Steam Games 2025')

model = joblib.load('model.joblib')

st.sidebar.image("https://cdn.who.int/media/images/default-source/infographics/who-emblem.png?sfvrsn=877bb56a_2", width=250)
st.sidebar.title('Steam Games 2025 App')
page = st.sidebar.radio("Navigate", ['Dashboard', 'ML'])
st.sidebar.markdown('---')
st.sidebar.caption('Built with Python and Streamlit')

if page == 'Dashboard':
    st.title('Steam Games Analytics Dashboard')
    st.markdown('Dashboard view is disabled since no data is loaded.')

elif page == "ML":
    st.title('Game Success Predictor')
    st.markdown('Predict if a game will be successful based on features')

    with st.expander('Game feature entry form', expanded=True):
        with st.form('prediction_form'):
            price = st.slider('Price', 0.0, 100.0, 10.0, step=0.5)
            dlc_count = st.slider('DLC Count', 0, 100, 0, step=1)

            metacritic_score = st.slider('Metacritic Score', 0, 100, 50, step=1)
            achievements = st.slider('Achievements', 0, 1000, 0, step=10)

            recommendations = st.slider('Recommendations', 0, 1000000, 0, step=1000)
            average_playtime_forever = st.slider('Average Playtime Forever', 0, 100000, 0, step=100)

            peak_ccu = st.slider('Peak Concurrent Users', 0, 1000000, 0, step=1000)
            avg_estimated_owners = st.slider('Average Estimated Owners', 0, 100000000, 0, step=10000)

            submit_btn = st.form_submit_button('Predict Success')

        if submit_btn:
            input_data = np.array([[
                price,
                dlc_count,
                metacritic_score,
                achievements,
                recommendations,
                average_playtime_forever,
                peak_ccu,
                avg_estimated_owners,
            ]])

            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0]

            # Debug info to verify inputs and model output
            st.write('Input features:', input_data)
            st.write('Prediction:', prediction)
            st.write('Probabilities:', proba)
            st.write('Model classes:', model.classes_)

            st.divider()

            col_r1, col_r2 = st.columns([1, 2])

            with col_r1:
                if prediction == 1:
                    st.success('High Success Probability')
                    conf = proba[1] 
                else:
                    st.error('Low Success Probability')
                    conf = proba[0] 

                st.metric('Confidence', f"{conf:.2%}")

            with col_r2:
                st.write('### Prediction Confidence')
                st.progress(conf)
                if conf > 0.5:
                    st.info('Game likely to be successful')
                else:
                    st.info('Game likely to underperform')

            # Additional test: extreme low input vs extreme high input
            if st.checkbox("Show Extreme Input Test"):
                min_input = np.array([[0,0,0,0,0,0,0,0]])
                max_input = np.array([[100,100,100,1000,1000000,100000,1000000,100000000]])
                st.write("Min Input Prediction:", model.predict(min_input)[0])
                st.write("Min Input Probabilities:", model.predict_proba(min_input)[0])
                st.write("Max Input Prediction:", model.predict(max_input)[0])
                st.write("Max Input Probabilities:", model.predict_proba(max_input)[0])
