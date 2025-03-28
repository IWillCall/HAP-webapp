import json
import numpy as np
import pandas as pd

with open('data/decode.json') as f:
    decode_sl = json.load(f)

columns = ['Sex', 'GeneralHealth', 'PhysicalHealthDays', 'MentalHealthDays',
     'PhysicalActivities', 'SleepHours', 'RemovedTeeth',
        'HadAsthma', 'HadSkinCancer', 'HadCOPD',
       'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis',
       'HadDiabetes', 'DeafOrHardOfHearing', 'BlindOrVisionDifficulty',
       'DifficultyConcentrating', 'DifficultyWalking',
       'DifficultyDressingBathing', 'DifficultyErrands', 'SmokerStatus',
       'ECigaretteUsage', 'ChestScan', 'AgeCategory', 'HeightInMeters',
       'WeightInKilograms', 'BMI', 'AlcoholDrinkers', 'HIVTesting',
       'FluVaxLast12', 'PneumoVaxEver', 'TetanusLast10Tdap',
       'HighRiskLastYear', 'CovidPos']



num_cols =['PhysicalHealthDays',
 'MentalHealthDays',
 'SleepHours',
 'HeightInMeters',
 'WeightInKilograms',
 'BMI']



def decode_request(request):
  request_sl = {}
  for col in columns:
    user_val = request.get(col)
    if user_val is None:
        request_sl[col] = np.nan
    if col in num_cols and col != 'BMI':
        request_sl[col] = float(user_val)
    else:
        response = decode_sl.get(user_val, user_val)
        response = np.nan if pd.isna(response) or response is None  or response=="NAN"  else response
        request_sl[col] = response
  request_sl["BMI"] = request_sl["WeightInKilograms"] / (request_sl["HeightInMeters"] ** 2)

  request_df = pd.DataFrame([request_sl])
  return request_df

def validate_request(form):
  with open("data/questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)
  errors = {}
  data = {}
  for key, q in questions.items():
      if q["input_type"] == "number":
          try:
              value = float(form.get(key, ""))
          except:
              errors[key] = "Невірне значення"
              continue
          if key == "HeightInMeters":
              if value <= 0 or value > 2.8:
                  errors[key] = "Невірне значення"
          elif key == "WeightInKilograms":
              if value <= 0 or value > 300:
                  errors[key] = "Невірне значення"
          elif key == "SleepHours":
              if value < 0 or value > 24:
                  errors[key] = "Невірне значення"
          elif key == "MentalHealthDays":
              if value < 0 or value > 31:
                  errors[key] = "Невірне значення"
          data[key] = value
      else:
          if q["input_type"] == "checkbox":
              val = form.get(key)
              data[key] = "Так" if val is not None else "Ні"
          else:
              data[key] = form.get(key, "")
  return {"errors": errors, "data": data}
