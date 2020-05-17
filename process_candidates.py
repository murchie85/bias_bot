"""
# Bias Recruitment AI
   
This code brings in the historical recruitment data (`train.csv`) and new candidate data (`test.csv`) then performs a decision/prediction on the new candidates if they will get the job or not. 

## Explanation 

Many AI algorythms are designed to automate jobs by doing it better/faster etc. In this case, the AI is making a decision based on the hiring history. I.e. If a company had shown bias towards/against a certain group then the AI would definitely pick up on this and possibly even do that even more. 

## Disclaimer

This is for those new to AI or not of a technical background.

## FLOW 

1. Import libs
2. Import CSVs
3. Set features and labels (information and values we wish to predict)
4. Build & Train model
5. Save results to CSV file 
"""




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


import os
for dirname, _, filenames in os.walk('data/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")


from sklearn.ensemble import RandomForestClassifier

y = train_data["hired"]

features = ["gender","disability"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)


predictions = model.predict(X_test)

output = pd.DataFrame({'candidateid': test_data.candidateid, 'gender':test_data.gender, 'disability':test_data.disability, 'hired': predictions})
output.to_csv('output.csv', index=False)
print("Model Ran successfully! Please open output.csv to view the data")