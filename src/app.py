import joblib
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from pathlib import Path
import random
import string
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocess import preprocess_data


model = joblib.load("../models/model.pkl")
titanic_train_df = pd.read_csv('../data/train.csv')
raw_data = titanic_train_df.copy()

titanic_train_df = preprocess_data (titanic_train_df, titanic_train_df)
# Maps

embarked = {
    "C": "Cherbourg",
    "Q": "Queenstown",
    "S": "Southampton"
}

ticket_class = {
    1: "1st",
    2: "2nd",
    3: "3rd"
}

survived = {
    0: "No",
    1: "Yes"
}

st.set_page_config(layout="wide")

st.title("Can You Survive The Titanic?")
st.caption("A machine learning project by Franciz Emmanuelle Angelo L. Esperanza | [Github](https://github.com/francizesperanza/titanic_survival_predictions)")

st.markdown('''
            For this project, we will take a look at the [**Titanic Survival Dataset**](https://www.kaggle.com/competitions/titanic/code)!
            We will look at the existing patterns between passenger characteristics and their survival rate.
            At the end, you will also be able to create a sample passenger and see whether they can survive or not.
            ''')

tab1, tab2, tab3 = st.tabs(["The Dataset", "Data Analysis", "Titanic Survival Simulator"])

with tab1:
    sample = raw_data.dropna().sample(n=1, random_state=69).reset_index(drop=True)
    st.header("The Dataset")
    st.markdown('''
                This dataset contains records of passengers who were on the Titanic the time it crashed on an iceberg.
                A single record contains their name, sex, age, ticket number and class, embarkation point, cabin number, the number of people they were with, and whether or not they survived the incident or not. 
                The `Survived` column is the label that we are trying to predict, where `0` is for Not Survived, and `1` is for Survived.

                In emergencies like this, women and children's safety are the utmost priority in evacuation measures.
                Upper class individuals also tend to have more chances of survival because of their social and financial advantages.
                ''')

    st.markdown("***Sample Passenger***")
    with st.container(border=True, horizontal_alignment="center", vertical_alignment="center", height="content"):
        col1, col2, col3, col4, col5, col6 = st.columns([1,1,1,1,1,1], vertical_alignment="center" , width="stretch")
        with col1:
            st.markdown("**Name**")
            st.write(sample.loc[0, "Name"])

            st.markdown("**Age**")
            st.write(sample.loc[0, "Age"].astype(int))
        with col2:
            st.markdown("**Sex**")
            st.write(sample.loc[0, 'Sex'])

            st.markdown("**Fare**")
            st.write(sample.loc[0, 'Fare'])

        with col3:
            st.markdown("**Ticket Number**")
            st.write(sample.loc[0, 'Ticket'])

            st.markdown("**Ticket Class**")
            st.write(ticket_class[sample.loc[0, 'Pclass']])

        with col4:
            st.markdown("**Embarked From**")
            st.write(embarked[sample.loc[0, 'Embarked']])

            st.markdown("**Cabin Number**")
            st.write(sample.loc[0, 'Cabin'])
        
        with col5:
            st.markdown("**Parents and Children Aboard**")
            st.write(sample.loc[0, 'Parch'])

            st.markdown("**Siblings and Spouses Aboard**")
            st.write(sample.loc[0, 'SibSp'])
        
        with col6:
            st.markdown(f"""
                        <div style="
                            background:#AFE1AF;
                            padding:16px;
                            border-radius:12px;
                        ">
                            <b>Survived</b><br>
                            {survived[sample.loc[0, 'Survived']]}
                        </div>
                        """, unsafe_allow_html=True)
with tab2:
    # Overall Survival Chart

    count_df = (
        titanic_train_df["Survived"]
        .value_counts()
        .rename_axis("Status")
        .reset_index(name="Count")
    )

    count_df["Status"] = count_df["Status"].map({0: "Dead", 1: "Alive"})

    # Title Chart

    title_df = titanic_train_df[["Title", "Survived"]].copy()

    title_survival = (
        titanic_train_df
        .groupby("Title", as_index=False)["Survived"]
        .mean().round(2)
        .rename(columns={"Survived": "SurvivalRate"})
        .sort_values(by="SurvivalRate")
    )

    title_survival['Title'] = title_survival['Title'].map({0: 'Mr.',1 : 'Miss', 2 : 'Mrs.',3: 'Master',4: 'Rare'})

    st.header("Data Analysis")

    with st.container():
        col1, col2 = st.columns([2,3], vertical_alignment="top" , width="stretch")
        with col1:
            with st.container(border=True, height="content"):
                st.markdown("**Survival Count**")
                st.caption("To no one's surprise, there were more passengers that died than those who survived.")
                st.bar_chart(count_df, x='Status', y='Count', color="Status", x_label="Passenger Status")
            
        with col2:
            with st.container(border=True, height="stretch"):
                st.markdown("**Survival Rate by Title**")
                st.caption("The `Title` feature tells us the age, sex, and class of a passenger in one word. Female passengers (Miss, Mrs.) have the highest survival rates, followed by young males (Master), professionals (Rare), and adult males (Mr). ")
                st.bar_chart(title_survival, x="Title", y="SurvivalRate", x_label="Passenger Title", y_label="Survival Rate", horizontal=True, color="Title", height=350)

    # Family Size Chart

    family_survival = (
        titanic_train_df
        .groupby("FamilySize", as_index=False)["Survived"]
        .mean().round(2)
        .rename(columns={"Survived": "SurvivalRate"})
        .sort_values(by="SurvivalRate")
    )

    with st.container():
        col1, col2 = st.columns([3,2], vertical_alignment="top" , width="stretch")
        with col1:
            with st.container(border=True, height="stretch"):
                st.markdown("**Survival Rate by Fare Per Person**")
                st.caption("There are higher survival counts in higher priced fares. This may indicate that those who can afford these tickets were prioritized during the evacuation.")
                fig = px.histogram(titanic_train_df, x="FarePerPerson", color="Survived", height=350)
                st.plotly_chart(fig)
            
        with col2:
            with st.container(border=True, height="stretch"):
                st.markdown("**Survival Rate by Family Size**")
                st.caption("`FamilySize` tells us how many people a person was travelling with, including themself. The trend shows us that lone travellers and large families (5-8) have a low survival rate, while small families have higher chances of living.")
                st.line_chart(family_survival, x="FamilySize", y="SurvivalRate", x_label="Family Size", y_label="Survival Rate", height=350)

    # Class Chart

    class_survival = (
        titanic_train_df
        .groupby("Pclass", as_index=False)["Survived"]
        .mean().round(2)
        .rename(columns={"Survived": "SurvivalRate"})
        .sort_values(by="SurvivalRate")
    )

    # Cabin Chart

    cabin_survival = (
        titanic_train_df
        .groupby("CabinCat", as_index=False)["Survived"]
        .mean().round(2)
        .rename(columns={"Survived": "SurvivalRate"})
        .sort_values(by="SurvivalRate")
    )

    cabin_survival["CabinCat"] = cabin_survival["CabinCat"].map({
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E",
        5: "F",
        6: "G",
        7: "U"
    })

    with st.container(vertical_alignment="top"):
        col1, col2 = st.columns([2,3], vertical_alignment="top" , width="stretch")
        with col1:
            with st.container(border=True):
                st.markdown("**Survival Rate by Cabin Category**")
                st.caption("The `CabinCat` column plot tells us that passengers with unknown (U) cabin numbers have a lower survival rate compared to those who have known cabin numbers. This may indicate that certain cabin areas are nearer to emergency exits or staircases.")
                fig = px.treemap(cabin_survival, path=[px.Constant("Cabin Category"), "CabinCat"], values='SurvivalRate', color="SurvivalRate", color_continuous_scale='RdYlGn', height=350)
                fig.update_layout(
                    treemapcolorway = ["pink"],
                    margin = dict(t=50, l=25, r=25, b=25)
                )
                fig.update_traces(
                    texttemplate="<b>%{label}</b><br>%{value:.0%}",
                    textfont_size=16
                )
                st.plotly_chart(fig)
        with col2:
            with st.container(border=True):
                st.markdown("**Survival Rate by Ticket Class**")
                st.caption("For the `Pclass` column, survival rate follows a downward trend as the ticket class descends. This indicates that higher class tickets are located in areas that are generally safer.")
                st.bar_chart(class_survival, x="Pclass", y="SurvivalRate", color="Pclass", x_label="Ticket Class", y_label="Survival Rate", height=350)

    with st.container():
        st.markdown("**Raw Data**")
        st.dataframe(titanic_train_df, width="stretch")

with tab3:
    st.header("Titanic Survivor Simulator")

    with st.container(border=True):
        
        st.markdown("**Personal Data**")
        col1, col2, col3 = st.columns(3)
        with col1:
            f_name = st.text_input("First Name")
            title = st.selectbox ("Title", ["Mr", "Mrs", "Miss", "Master", "Dr", "Rev", "Dona", "Sir", "Major"])
        with col2:
            l_name = st.text_input("Last Name")
            parch = st.number_input("No. of Parents / Children Travelling With", step=1, min_value=0)

        with col3:
            colA, colB = st.columns(2)
            with colA:
                sex = st.selectbox("Sex", ["Male", "Female"])
            with colB:
                age = st.slider("Age", 0, 130, 25)
            sibsp = st.number_input("No. of Siblings / Spouses Travelling With", step=1, min_value=0)
    
    predict = False
    with st.container(border=True):
        
        st.markdown("**Ticketing Data**")
        col1, col2, col3 = st.columns(3)

        with col1:
            p_class = st.selectbox("Ticket Class", ["1st", "2nd", "3rd"])
            ticket_fare = st.number_input("Ticket Fare", value=25.0, min_value=0.0)
        with col2:
            cabin = st.selectbox("Cabin Type", ["A", "B", "C", "D", "E", "F", "G", "N/A"])
            ticket_number = cabin + ''.join(random.choices(string.digits, k=6))
            cabin = {"N/A": np.nan}.get(cabin, cabin)
            st.text_input("Ticket Number", value=ticket_number, disabled=True)
        with col3:
            embarked = st.selectbox("Embarking From", ["Cherbourg", "Queenstown", "Southampton"])
            with st.container(vertical_alignment="bottom", horizontal_alignment="right", width="stretch", height="stretch"):
                if st.button("Book Ticket", type="primary", icon="🛳️"):
                    predict = True
        
    if predict:

        new_row = pd.DataFrame({
            "PassengerId": [0],
            "Pclass": [p_class[0]],
            "Name": [l_name + ', ' + title + '. ' + f_name],
            "Sex": [sex.lower()],
            "Age": [age],
            "SibSp": [sibsp],
            "Parch": [parch],
            "Ticket": [ticket_number],
            "Fare": [ticket_fare],
            "Cabin": [cabin],
            "Embarked": [embarked[0]]
        })

        print(new_row)

        prediction = model.predict(new_row)

        if prediction[0] == 1:
            st.markdown(f"""
                    <div style="
                        background:#AFE1AF;
                        padding:16px;
                        border-radius:12px;
                        font-size: 1em;
                    ">
                        <h1><b>🛟Congratulations!</b></h1>
                        <div>You survived the Titanic incident. Enjoy your trip...</div>
                    </div>
                    """, unsafe_allow_html=True)
        elif prediction[0] == 0:
            st.markdown(f"""
                    <div style="
                        background:#FAA0A0;
                        padding:16px;
                        border-radius:12px;
                        font-size: 1em;
                    ">
                        <h1><b>💀Unfortunate.</b></h1>
                        <div>You died in the Titanic incident. Do not book this ticket.</div>
                    </div>
                    """, unsafe_allow_html=True)
