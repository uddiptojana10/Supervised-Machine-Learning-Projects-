import streamlit as st
import pickle
import pandas as pd

# Teams and cities
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
         'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
         'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

# Load trained model
pipe = pickle.load(open('pipe.pkl', 'rb'))

st.title('🏏 IPL Win Predictor')

# Team selection
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    filtered_teams = [team for team in teams if team != batting_team]
    bowling_team = st.selectbox('Select the bowling team', sorted(filtered_teams))

# City selection
selected_city = st.selectbox('Select host city', sorted(cities))

# Target input
target = st.number_input('Target', min_value=1, step=1, format="%d")

# Match stats
col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Score', min_value=0, step=1, format="%d")
with col4:
    valid_overs = [round(x * 0.1, 1) for x in range(0, 201) if round((x * 0.1) % 1, 1) in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]]
    overs = st.selectbox('Overs completed', options=valid_overs, index=valid_overs.index(0.0))  # Default to 0.0
with col5:
    wickets_out = st.number_input('Wickets out', min_value=0, max_value=10, step=1, format="%d")

# Helper: check if overs format is valid (.0 to .5 only)
def is_valid_over(over):
    decimal = round(over % 1, 1)
    return decimal in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# Prediction logic
if st.button('Predict Probability'):
    if not is_valid_over(overs):
        st.error("Overs can only be up to .5 — e.g., 18.0, 18.1, ..., 18.5")
    else:
        runs_left = target - score
        balls_left = 120 - int(overs * 6)
        wickets = 10 - wickets_out
        crr = score / overs if overs > 0 else 0  # Avoid division by zero
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        # Logical cricket overrides
        if wickets_out >= 10:
            win = 0.0
            loss = 1.0
            st.warning("All wickets are down. Match is over.")
        elif score >= target:
            win = 1.0
            loss = 0.0
            st.success("Target achieved! Batting team wins.")
        elif runs_left <= 1 and balls_left > 0:
            win = 1.0
            loss = 0.0
            st.success("Just 1 run needed with balls left — almost certain win.")
        elif balls_left <= 0 and score < target:
            win = 0.0
            loss = 1.0
            st.warning("No balls left and target not reached. Bowling team wins.")
        else:
            # Predict with model
            input_df = pd.DataFrame({
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'city': [selected_city],
                'runs_left': [runs_left],
                'balls_left': [balls_left],
                'wickets': [wickets],
                'total_runs_x': [target],
                'crr': [crr],
                'rrr': [rrr]
            })

            result = pipe.predict_proba(input_df)
            loss = result[0][0]
            win = result[0][1]

        st.subheader("📊 Win Probability")
        st.header(f"{batting_team}: {round(win * 100)}%")
        st.header(f"{bowling_team}: {round(loss * 100)}%")

        # Optional: summary line
        if balls_left > 0:
            st.markdown(f"**{runs_left} runs needed in {balls_left} balls with {wickets} wickets in hand.**")