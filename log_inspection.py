import pandas as pd

df = pd.read_csv('./data/loan_baseline.pnml_noise_0.15_iteration_1_seed_614_simple.csv')

groups = df.groupby('Case ID')

first_event = {}
total_e = 0
for _, group in groups:
    first_e = list(group['Activity'])[1]
    total_e+=1
    if first_e not in list(first_event.keys()):
        first_event[first_e]= 1
    else:
        first_event[first_e] += 1

for t in first_event:
    print(t, 100 * first_event[t]/total_e)
# print(first_event)