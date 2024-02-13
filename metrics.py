import pandas as pd
from sklearn.metrics import accuracy_score
data = pd.read_excel('./predict_correct.xlsx')
accuracy = accuracy_score(data['correct'], data['predict'])
print(accuracy)

correlation = data['predict'].corr(data['correct'])
print(correlation)
