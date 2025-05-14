import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer # import imputer
from sklearn.model_selection import RandomizedSearchCV
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

def stratified_c_index(estimator, X, y):
    """
    Calculate the Stratified C-index, adjusted for racial stratification.

    Parameters:
    - estimator: The fitted model (survival estimator).
    - X: The input features.
    - y: The target data (must contain both time and event columns).
    - race_column: The column in `X` representing the race category.

    Returns:
    - stratified_c_index: The stratified concordance index.
    """

    # Initialize a list to store the C-index values for each racial group
    c_index_values = []

    # Loop through each unique race category and compute the C-index for each group
    for race in X["RIDRETH1"].unique():
        # Filter data for this racial group
        X_race = X[X["RIDRETH1"] == race]
        y_race = y[X["RIDRETH1"] == race]

        # Predict survival function (or predict risk, depending on your estimator)
        y_pred = estimator.predict(X_race)

        # Calculate C-index for this race category
        c_index = concordance_index_censored(y_race["mortstat"], y_race["permth_int"], y_pred)

        c_index_values.append(c_index[0])  # C-index is the first element of the tuple

    # Calculate mean and standard deviation of C-index values
    mean_c_index = np.mean(c_index_values)
    std_c_index = np.std(c_index_values)

    # Stratified C-index is the mean minus the standard deviation
    stratified_c_index = mean_c_index - std_c_index

    return stratified_c_index
# Streamlit App
st.set_page_config(layout="wide")
st.title("Survival Outcome Prediction")
st.write("Despite the growing recognition of social determinants of health (SDOH) in shaping health outcomes, predictive models that integrate these factors while ensuring fairness across racial groups remain scarce. We have developed a machine-learning model incorporating SDOH for predicting all-cause mortality in the general population, ensuring equitable performance across races. We have translated the developed model onto this application where users can input patient data—including demographic characteristics, social determinants of health, medical history, laboratory results, and physical examination findings—to generate individualized survival curves and survival probability estimates for up to 20 years.")
# Collect user input
st.subheader("Social and demographic data")
col1, col2, col3 = st.columns(3)
with col1:
    RIAGENDR_mapping = {"Male": 1.0, "Female": 2.0, "Select": np.nan}
    RIAGENDR = st.selectbox("Gender", options=["Select", "Male", "Female"])
    RIAGENDR = RIAGENDR_mapping[RIAGENDR]
    RIDAGEYR = st.slider("Age in years", 18, 85, 47)  # min: 18, max: 85, default: 47 (median)
    DMDCITZN_mapping = {"Citizen": 1.0, "Not a citizen of the US": 2.0, "Select": np.nan}
    DMDCITZN = st.selectbox("Are you a citizen of the United States?", options=["Select", "Citizen", "Not a citizen of the US"])
    DMDCITZN = DMDCITZN_mapping[DMDCITZN]
    DMDEDUC2_mapping = {"Less Than 9th Grade": 1, "9-11th Grade including 12th grade with no diploma": 2, "High School Grad or GED or Equivalent": 3, "Some College or AA degree": 4, "College Graduate or above": 5}
    DMDEDUC2 = st.selectbox("What is the highest grade or level of school you have completed or the highest degree you have received?", options=["Less Than 9th Grade", "9-11th Grade including 12th grade with no diploma", "High School Grad or GED or Equivalent", "Some College or AA degree", "College Graduate or above"])
    DMDEDUC2 = DMDEDUC2_mapping[DMDEDUC2]
    DMDMARTL_mapping = {"Married": 1, "Widowed": 2, "Divorced": 3, "Separated": 4, "Never married": 5, "Living with partner": 6}
    DMDMARTL = st.selectbox("Marital Status", options=["Married", "Widowed", "Divorced", "Separated", "Never married", "Living with partner"])
    DMDMARTL = DMDMARTL_mapping[DMDMARTL]
    DMQMILIT_mapping = {"Yes": 1, "No": 2}
    DMQMILIT = st.selectbox("Military Status", options=["Yes", "No"])
    DMQMILIT = DMQMILIT_mapping[DMQMILIT]
    DMDHHSIZ = st.slider("Total number of people in the Household", 1, 7, 3, step = 1, help = "if there are 7 or more people in the Household, put 7")
    RIDRETH1_mapping = {"Mexican American": 1, "Other Hispanic": 2, "Non-Hispanic White": 3, "Non-Hispanic Black": 4, "Other Race including Multi-Racial": 5}
    RIDRETH1 = st.selectbox("Race or Ethnicity", options=["Mexican American", "Other Hispanic", "Non-Hispanic White", "Non-Hispanic Black", "Other Race including Multi-Racial"])
    RIDRETH1 = RIDRETH1_mapping[RIDRETH1]
    
with col2:    
    FSDAD_mapping = {"full food security 0": 1, "marginal food security 1 or 2": 2, "low food security 3 to 5": 3, "very low food security 6 to 10": 4}
    FSDAD = st.selectbox("Adult food security category for last 12 months", options=["full food security 0", "marginal food security 1 or 2", "low food security 3 to 5", "very low food security 6 to 10"])
    FSDAD = FSDAD_mapping[FSDAD]
    INDFMPIR = st.slider("Poverty income ratio as a ratio of family income to poverty threshold", 0.0, 5.0, value=2.06, step=0.01, help = "Select 5 if PIR value is greater than or equal to 5.00")
    FSDHH_mapping = {"full food security 0": 1, "marginal food security 1 or 2": 2, "low food security 3 to 5 or 3 to 7 for HH with child": 3, "very low food security 6 to 10 or 8 to 18 for HH with child": 4}
    FSDHH = st.selectbox("Household food security category for last 12 months", options=["full food security 0", "marginal food security 1 or 2", "low food security 3 to 5 or 3 to 7 for HH with child", "very low food security 6 to 10 or 8 to 18 for HH with child"])
    FSDHH = FSDHH_mapping[FSDHH]
    HIQ011_mapping = {"Yes": 1, "No": 2}
    HIQ011 = st.selectbox("Covered by health insurance", options=["Yes", "No"])
    HIQ011 = HIQ011_mapping[HIQ011]
    HUQ010_mapping = {"Excellent": 1, "Very good": 2, "Good": 3, "Fair": 4, "Poor": 5}
    HUQ010 = st.selectbox("Would you say your health in general is", options=["Excellent", "Very good", "Good", "Fair", "Poor"])
    HUQ010 = HUQ010_mapping[HUQ010]
    HUQ030_mapping = {"Yes": 1, "There is no place": 2, "There is more than one place": 3}
    HUQ030 = st.selectbox("Is there a place that you usually go when you are sick or you need advice about your health", options=["Yes", "There is no place", "There is more than one place"])
    HUQ030 = HUQ030_mapping[HUQ030]
    HUQ040_mapping = {"Clinic or health center": 1, "Doctor's office or HMO": 2, "Hospital emergency room": 3, "Hospital outpatient department": 4, "Some other place": 5}
    HUQ040 = st.selectbox("Type place most often go for healthcare", options=["Clinic or health center", "Doctor's office or HMO", "Hospital emergency room", "Hospital outpatient department", "Some other place"])
    HUQ040 = HUQ040_mapping[HUQ040]
    HUQ050_mapping = {"None": 0, "1": 1, "2 to 3": 2, "4 to 9": 3, "10 to 12": 4, "13 or more": 5}
    HUQ050 = st.selectbox("Times receive healthcare over past year", options=["None", "1", "2 to 3", "4 to 9", "10 to 12", "13 or more"])
    HUQ050 = HUQ050_mapping[HUQ050]
    
with col3: 
    HUQ071_mapping = {"Yes": 1, "No": 2}
    HUQ071 = st.selectbox("Overnight hospital patient in last year", options=["Yes", "No"])
    HUQ071 = HUQ071_mapping[HUQ071]
    HOQ065_mapping = {"Owned or being bought": 1, "Rented": 2, "Other arrangement": 3}
    HOQ065 = st.selectbox("Home owned, bought, rented, other", options=["Owned or being bought", "Rented", "Other arrangement"])
    HOQ065 = HOQ065_mapping[HOQ065]
    HOD050 = st.slider("How many rooms are in this home Count the kitchen but not the bathroom", 1, 13, 6, step=1, help="Select 13 if 13 or More")  # min: 1, max: 12, default: 6 (mean)
    OCD150_mapping = {"Working at a job or business": 1, "With a job or business but not at work": 2, "Looking for work": 3, "Not working at a job or business": 4}
    OCD150 = st.selectbox("Type of work done last week", options=["Working at a job or business", "With a job or business but not at work", "Looking for work", "Not working at a job or business"])
    OCD150 = OCD150_mapping[OCD150]
    PFQ090_mapping = {"Yes": 1, "No": 2}
    PFQ090 = st.selectbox("Do you now have any health problem that requires you to use special equipment like a cane, a wheelchair, a special bed, or a special telephone", options=["Yes", "No"])
    PFQ090 = PFQ090_mapping[PFQ090]
    DMDBORN4_mapping = {"Born in 50 US states or Washington, DC": 1, "Others": 2}
    DMDBORN4 = st.selectbox("Country of birth", options=["Born in 50 US states or Washington, DC", "Others"])
    DMDBORN4 = DMDBORN4_mapping[DMDBORN4]
    DMDHRAGZ_mapping = {"<20 years": 1, "20-39 years": 2, "40-59 years": 3, "60+ years": 4}
    DMDHRAGZ = st.selectbox("Household reference person age in years", options=["<20 years", "20-39 years", "40-59 years", "60+ years"])
    DMDHRAGZ = DMDHRAGZ_mapping[DMDHRAGZ]
    DMDHREDZ_mapping = {"Less than high school degree": 1, "High school grad/GED or some college/AA degree": 2, "College graduate or above": 3}
    DMDHREDZ = st.selectbox("Household reference person education level", options=["Less than high school degree", "High school grad/GED or some college/AA degree", "College graduate or above"])
    DMDHREDZ = DMDHREDZ_mapping[DMDHREDZ]
    DMDHRMAZ_mapping = {"Married/Living with partner": 1, "Widowed/Divorced/Separated": 2, "Never Married": 3}
    DMDHRMAZ = st.selectbox("Household reference person marital status", options=["Married/Living with partner", "Widowed/Divorced/Separated", "Never Married"])
    DMDHRMAZ = DMDHRMAZ_mapping[DMDHRMAZ]
    
st.subheader("Medical history")
colu1, colu2, colu3 = st.columns(3)
with colu1: 
    DIQ010_mapping = {"Yes": 1, "No": 2, "Borderline": 3}
    DIQ010 = st.selectbox("Doctor told you have diabetes", options=["Yes", "No", "Borderline"])
    DIQ010 = DIQ010_mapping[DIQ010]
    DIQ050_mapping = {"Yes": 1, "No": 2}
    DIQ050 = st.selectbox("Taking insulin now", options=["Yes", "No"])
    DIQ050 = DIQ050_mapping[DIQ050]
    KIQ022_mapping = {"Yes": 1, "No": 2}
    KIQ022 = st.selectbox("Ever told you had weak or failing kidneys", options=["Yes", "No"])
    KIQ022 = KIQ022_mapping[KIQ022]
    MCQ053_mapping = {"Yes": 1, "No": 2}
    MCQ053 = st.selectbox("Taking treatment for anemia in the past 3 mos", options=["Yes", "No"])
    MCQ053 = MCQ053_mapping[MCQ053]
    MCQ092_mapping = {"Yes": 1, "No": 2}
    MCQ092 = st.selectbox("Ever receive blood transfusion", options=["Yes", "No"])
    MCQ092 = MCQ092_mapping[MCQ092]
    MCQ160A_mapping = {"Yes": 1, "No": 2}
    MCQ160A = st.selectbox("Doctor ever said you had arthritis", options=["Yes", "No"])
    MCQ160A = MCQ160A_mapping[MCQ160A]
        
with colu2: 
    MCQ160B_mapping = {"Yes": 1, "No": 2}
    MCQ160B = st.selectbox("Ever told had congestive heart failure", options=["Yes", "No"])
    MCQ160B = MCQ160B_mapping[MCQ160B]
    MCQ160C_mapping = {"Yes": 1, "No": 2}
    MCQ160C = st.selectbox("Ever told you had coronary heart disease", options=["Yes", "No"])
    MCQ160C = MCQ160C_mapping[MCQ160C]
    MCQ160D_mapping = {"Yes": 1, "No": 2}
    MCQ160D = st.selectbox("Ever told you had angina pectoris", options=["Yes", "No"])
    MCQ160D = MCQ160D_mapping[MCQ160D]
    MCQ160E_mapping = {"Yes": 1, "No": 2}
    MCQ160E = st.selectbox("Ever told you had heart attack", options=["Yes", "No"])
    MCQ160E = MCQ160E_mapping[MCQ160E]
    MCQ160F_mapping = {"Yes": 1, "No": 2}
    MCQ160F = st.selectbox("Ever told you had a stroke", options=["Yes", "No"])
    MCQ160F = MCQ160F_mapping[MCQ160F]
    MCQ160G_mapping = {"Yes": 1, "No": 2}
    MCQ160G = st.selectbox("Ever told you had emphysema", options=["Yes", "No"])
    MCQ160G = MCQ160G_mapping[MCQ160G]
    
with colu3: 
    MCQ160K_mapping = {"Yes": 1, "No": 2}
    MCQ160K = st.selectbox("Ever told you had chronic bronchitis", options=["Yes", "No"])
    MCQ160K = MCQ160K_mapping[MCQ160K]
    MCQ160L_mapping = {"Yes": 1, "No": 2}
    MCQ160L = st.selectbox("Ever told you had any liver condition", options=["Yes", "No"])
    MCQ160L = MCQ160L_mapping[MCQ160L]
    MCQ220_mapping = {"Yes": 1, "No": 2}
    MCQ220 = st.selectbox("Ever told you had cancer or malignancy", options=["Yes", "No"])
    MCQ220 = MCQ220_mapping[MCQ220]
    MCQ160M_mapping = {"Yes": 1, "No": 2}
    MCQ160M = st.selectbox("Ever told you had a thyroid problem", options=["Yes", "No"])
    MCQ160M = MCQ160M_mapping[MCQ160M]
    SMQ020_mapping = {"Yes": 1, "No": 2}
    SMQ020 = st.selectbox("Have you smoked at least 100 cigarettes in your entire life", options=["Yes", "No"])
    SMQ020 = SMQ020_mapping[SMQ020]
    WHQ070_mapping = {"Yes": 1, "No": 2}
    WHQ070 = st.selectbox("During the past 12 months, have you tried to lose weight", options=["Yes", "No"])
    WHQ070 = WHQ070_mapping[WHQ070]
    BPGHYP_mapping = {"Yes": 1, "No": 2}
    BPGHYP = st.selectbox("Ever told you have hypertension or took antihypertensive drugs", options=["Yes", "No"])
    BPGHYP = BPGHYP_mapping[BPGHYP]

st.subheader("Physical examination data")
colum1, colum2 = st.columns(2)
with colum1:
    BPXPULS_mapping = {"Regular": 1, "Irregular": 2}
    BPXPULS = st.selectbox("Pulse regular or irregular", options=["Regular", "Irregular"])
    BPXPULS = BPXPULS_mapping[BPXPULS]
    BPXPLS = st.slider("60 sec. pulse", 32, 220, 72)
    BMXBMI = st.slider("Body Mass Index", 12.0, 50.0, 27.6)
with colum2:    
    BMXWAIST = st.slider("Waist Circumference (cm)", 55.0, 180.0, 96.6)
    BPXSY_avg = st.slider("Average systolic blood pressure mmHg", 60.0, 270.0, 121.33)
    BPXDI_avg = st.slider("Average diasolic blood pressure mmHg", 10.0, 140.0, 70.7)

st.subheader("Laboratory investigation data")
co1, co2, co3 = st.columns(3)
with co1:
    URXUCR = st.slider("Creatinine, urine (mg/dL)", 3.0, 900.0, 115.0)
    URXUMA = st.slider("Albumin, urine (ug/mL)", 0.2, 24440.0, 8.3)
    LBDMONO = st.slider("Monocyte number", 0.0, 6.0, 0.5)
    LBDNENO = st.slider("Segmented neutrophils number", 0.0, 84.0, 4.0)
    LBXHGB = st.slider("Hemoglobin (g/dL)", 5.0, 20.0, 14.2)
    LBXMCHSI = st.slider("Mean cell hemoglobin (pg)", 14.0, 75.0, 30.5)
    LBXMCVSI = st.slider("Mean red cell volume (fL)", 50.0, 130.0, 89.7)
    LBXPLTSI = st.slider("Platelet count (%) SI", 8.0, 1000.0, 246.0)
    LBXRBCSI = st.slider("Red cell count SI", 1.0, 10.0, 4.68)
with co2:
    LBXRDW = st.slider("Red cell distribution width (%)", 10.0, 40.0, 12.9)
    LBXWBCSI = st.slider("White blood cell count (SI)", 1.0, 400.0, 6.90)
    LBDHDDSI = st.slider("Direct HDL-Cholesterol (mmol/L)", 0.1, 6.0, 1.29)
    LBXGH = st.slider("Glycohemoglobin(%)", 2.0, 19.0, 5.5)
    LBDSALSI = st.slider("Albumin (g/L)", 12.0, 60.0, 43.0)
    LBXSAPSI = st.slider("Alkaline phosphotase (U/L)", 7.0, 750.0, 68.0)
    LBDSBUSI = st.slider("Blood urea nitrogen (mmol/L)", 0.3, 45.0, 4.64)
    LBDSCHSI = st.slider("Cholesterol, total (mmol/L)", 1.0, 25.0, 4.91)
with co3:
    LBDSGLSI = st.slider("Glucose (mmol/L)", 1.0, 40.0, 5.11)
    LBXSLDSI = st.slider("Lactate dehydrogenase LDH (U/L)", 30.0, 1300.0, 131.0)
    LBDSTRSI = st.slider("Triglycerides (mmol/L)", 0.0, 70.0, 1.31)
    LBDSUASI = st.slider("Uric acid (umol/L)", 20.0, 1100.0, 315.2)
    LBDSCRSI = st.slider("Creatinine (umol/L)", 15.0, 1600.0, 74.26)
    LBXSKSI = st.slider("Potassium (mmol/L)", 2.0, 7.5, 4.0)
    LBXSCLSI = st.slider("Chloride (mmol/L)", 70.0, 120.0, 103.10)
    LBXSOSSI = st.slider("Osmolality (mOsm/kg)", 200.0, 325.0, 278.0)

categorical_features = {
    "DMDCITZN": DMDCITZN,
    "DMDEDUC2": DMDEDUC2,
    "DMDMARTL": DMDMARTL,
    "DMQMILIT": DMQMILIT,
    "RIAGENDR": RIAGENDR,
    "RIDRETH1": RIDRETH1,
    "DIQ010": DIQ010,
    "DIQ050": DIQ050,
    "FSDAD": FSDAD,
    "FSDHH": FSDHH,
    "HIQ011": HIQ011,
    "HUQ010": HUQ010,
    "HUQ030": HUQ030,
    "HUQ040": HUQ040,
    "HUQ050": HUQ050,
    "HUQ071": HUQ071,
    "HOQ065": HOQ065,
    "KIQ022": KIQ022,
    "MCQ053": MCQ053,
    "MCQ092": MCQ092,
    "MCQ160A": MCQ160A,
    "MCQ160B": MCQ160B,
    "MCQ160C": MCQ160C,
    "MCQ160D": MCQ160D,
    "MCQ160E": MCQ160E,
    "MCQ160F": MCQ160F,
    "MCQ160G": MCQ160G,
    "MCQ160K": MCQ160K,
    "MCQ160L": MCQ160L,
    "MCQ220": MCQ220,
    "MCQ160M": MCQ160M,
    "OCD150": OCD150,
    "PFQ090": PFQ090,
    "SMQ020": SMQ020,
    "WHQ070": WHQ070,
    "BPXPULS": BPXPULS,
    "DMDBORN4": DMDBORN4,
    "DMDHRAGZ": DMDHRAGZ,
    "DMDHREDZ": DMDHREDZ,
    "DMDHRMAZ": DMDHRMAZ,
    "BPGHYP": BPGHYP
}

continuous_features = {
    "DMDHHSIZ": DMDHHSIZ,
    "INDFMPIR": INDFMPIR, 
    "RIDAGEYR": RIDAGEYR,
    "HOD050": HOD050, 
    "URXUCR": URXUCR,
    "URXUMA": URXUMA,
    "LBDMONO": LBDMONO,
    "LBDNENO": LBDNENO,
    "LBXHGB": LBXHGB,
    "LBXMCHSI": LBXMCHSI,
    "LBXMCVSI": LBXMCVSI,
    "LBXPLTSI": LBXPLTSI,
    "LBXRBCSI": LBXRBCSI,
    "LBXRDW": LBXRDW, 
    "LBXWBCSI": LBXWBCSI,
    "LBDHDDSI": LBDHDDSI,
    "LBXGH": LBXGH,
    "LBDSALSI": LBDSALSI,
    "LBXSAPSI": LBXSAPSI, 
    "LBDSBUSI": LBDSBUSI,
    "LBDSCHSI": LBDSCHSI,
    "LBDSGLSI": LBDSGLSI, 
    "LBXSLDSI": LBXSLDSI, 
    "LBDSTRSI": LBDSTRSI,
    "LBDSUASI": LBDSUASI,
    "LBDSCRSI": LBDSCRSI, 
    "LBXSKSI": LBXSKSI, 
    "LBXSCLSI": LBXSCLSI,
    "LBXSOSSI": LBXSOSSI, 
    "BPXPLS": BPXPLS,
    "BMXBMI": BMXBMI,
    "BMXWAIST": BMXWAIST,
    "BPXSY_avg": BPXSY_avg, 
    "BPXDI_avg": BPXDI_avg
}

data = pd.DataFrame({**categorical_features, **continuous_features}, index=[0])

# Load the trained model
model = joblib.load("random_search_COX_model_stratified.pkl")
# Prediction
if st.button("Predict"):
    # Ensure the dataframe is Arrow-compatible by casting data types
    data = data.astype({col: "float" for col in data.columns})
    try:
        prediction = model.best_estimator_.predict_survival_function(data)
        # Extract survival probabilities and times
        survival_times = prediction[0].x  # Time points
        survival_probs = prediction[0].y  # Survival probabilities
        
        # Create a table of time and survival probabilities
        survival_df = pd.DataFrame({
            "Time in months": survival_times,
            "Survival Probability": survival_probs
        })
        
        # Display the survival table
        st.subheader("Survival Predictions")
        column1, column2 = st.columns(2)
        with column1:    
            st.write(survival_df)
        
        with column2:
        # Plot survival curve
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(survival_times, survival_probs, label="Survival Curve", color="green")
            plt.xlabel("Time, months")
            plt.ylabel("Survival Probability")
            plt.title("Survival Curve fo your patient")
            plt.legend()
            plt.grid(True)
        
            # Display plot in Streamlit
            st.pyplot(plt)
    except Exception as e:
        st.error(f"An error occurred: {e}")

