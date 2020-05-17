# BIAS RECRUITMENT AI

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


## INSTRUCTIONS

If you have python & pip installed you need to run the following commands to install dependencies
```
pip install numpy
pip install pandas
pip install sklearn 
```  

  
Then in terminal or command line navigate to the folder where this code is and run the following: 
```
python process_candidates.py
```  
This will create the output csv that you can view. 

## CHALLENGE 

Change the train.csv and add more bias, imagine what sort of bias may be going on before the use of AI, then run the program and see if the AI picks up on that underlying bias. 