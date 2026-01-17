import gradio as gr
import pandas as pd
import pickle
import numpy as np

#====================
#Load the trained model
#====================
with open('student_rf_pipeline.pkl', 'rb') as f:
    model = pickle.load(f)
    
# =====================
#main logic
# =====================

def predict_gpa(gender,age,address,famsize,Pstatus,M_Edu,F_Edu,
                M_Job,F_Job,relationship,smoker,
                tuition_fee,time_friends,ssc_result):
    input_df = pd.DataFrame([[
        gender,age,address,famsize,Pstatus,M_Edu,F_Edu,
        M_Job,F_Job,relationship,smoker,
        tuition_fee,time_friends,ssc_result
    ]], 
    columns=[
        'gender','age','address','famsize','Pstatus','M_Edu','F_Edu',
        'M_Job','F_Job','relationship','smoker',
        'tuition_fee','time_friends','ssc_result'
    ])
    
    #prediction
    prediction = model.predict(input_df)[0]
    return f"Predicted HSC GPA: {np.clip(prediction, 0, 5):.2f}"
inputs = [
    gr.Radio(["M","F"], label="Gender"),
    gr.Number(label="Age",value=17),
    gr.Radio(["Urban","Rural"], label="Address"),
    gr.Radio(["LE3","GT3"], label="Family Size"),
    gr.Radio(["Together","Apart"], label="Parent's Cohabitation Status"),  
    gr.Slider(0,4, step=1, label="Mother's Education Level", elem_id="M_Edu"),
    gr.Slider(0,4, step=1, label="Father's Education Level", elem_id="F_Edu"),
    gr.Dropdown(["Teacher","Health","Services","At_home","Other"], label="Mother's Job", elem_id="M_Job"),
    gr.Dropdown(["Teacher","Health","Services","Business","Farmer","Other"], label="Father's Job", elem_id="F_Job"),
    gr.Radio(["Yes", "No"], label="Relationship"),
    gr.Radio(["Yes", "No"], label="Smoker"),
    gr.Number(label="Tuition Fee", elem_id="tuition_fee"),
    gr.Slider(1, 5, step=1, label="Time with Friends", elem_id="time_friends"),
    gr.Number(label="SSC Result (GPA)")
]
    

# =====================
#interface design
# =====================
app = gr.Interface(
    fn=predict_gpa,
    inputs=inputs,
    outputs=("text"),
    title="Bangladesh Student HSC GPA Prediction",
)
# =====================
#app launch
# =====================
app.launch()