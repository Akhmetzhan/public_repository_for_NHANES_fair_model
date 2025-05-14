This is a public repository for the paper titled "Fair Mortality Prediction Across Racial Groups in General Population: A Machine Learning Model Integrating Social Determinants and Clinical Risk Factors Using NHANES Data"

Abstract
Background Despite the growing recognition of social determinants of health (SDOH) in shaping health outcomes, predictive models that integrate these factors while ensuring fairness across racial groups remain scarce. 

Aim This study aimed to develop a machine learning model (ML) incorporating SDOH for predicting all-cause mortality in the general population, ensuring equitable performance across races.

Methods We utilized data from the National Health and Nutrition Examination Survey (NHANES) from 1999 to 2018, linked to mortality records through 2019. A weighted population of 10,250,542,924 participants were analyzed, incorporating 75 variables spanning SDOH, medical history, and laboratory results. Five ML algorithms were evaluated: Cox proportional hazards, Coxnet, survival tree, random survival forest, and gradient boosting. Model performance was assessed using the stratified concordance index (C-index) across racial groups.

Results The Cox proportional hazards model exhibited the best balance of discrimination (C-index: 0.88, 95% CI: 0.86–0.89) and calibration, with an integrated Brier score of 0.069. Adding SDOH significantly improved the predictive power and calibration of the model. Although the gradient boosting model achieved comparable discrimination, its calibration was suboptimal. The Cox model was deployed as an open-source Streamlit web application, allowing users to input patient data and generate 20-year survival predictions. Among the key predictors were age, gender, poor self-rated health, and low income.

Conclusion This study underscores the significance of incorporating SDOH into predictive models and introduces a practical, transparent tool for individualized mortality risk assessment.

Keywords: machine learning, fair learning, social determinants of health, survival analysis, mortality.
