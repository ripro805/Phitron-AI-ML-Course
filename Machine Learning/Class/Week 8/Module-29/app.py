import gradio as gr
import pandas as pd
import pickle

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Feature order (important)
feature_columns = model.feature_names_in_

def predict_attrition(
    Age,
    MonthlyIncome,
    TotalWorkingYears,
    YearsAtCompany,
    OverTime
):
    # Create input dataframe
    input_dict = {col: 0 for col in feature_columns}

    input_dict["Age"] = Age
    input_dict["MonthlyIncome"] = MonthlyIncome
    input_dict["TotalWorkingYears"] = TotalWorkingYears
    input_dict["YearsAtCompany"] = YearsAtCompany

    if OverTime == "Yes":
        input_dict["OverTime_Yes"] = 1

    input_df = pd.DataFrame([input_dict])

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        return f"Employee is likely to LEAVE (Attrition)\nProbability: {prob:.2f}"
    else:
        return f"Employee is likely to STAY\nProbability: {prob:.2f}"

# Gradio UI
interface = gr.Interface(
    fn=predict_attrition,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="Monthly Income"),
        gr.Slider(label="Total Working Years"),
        gr.Slider(label="Years At Company"),
        gr.Radio(["Yes", "No"], label="OverTime")
    ],
    outputs="text",
    title="Employee Attrition Prediction System",
    description="Predict whether an employee is likely to leave the company using ML"
)

if __name__ == "__main__":
    interface.launch()
