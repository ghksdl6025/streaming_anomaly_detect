import pandas as pd

df = pd.read_csv('./data/loan_baseline.pnml_noise_0.0_iteration_1_seed_98070_sample.csv')

groups = df.groupby('Case ID')

concating = []
for _, group in groups:
    group = group.reset_index(drop=True)
    start_activity ='Start'
    cid = _
    start_ts = list(group['Complete Timestamp'])[0]
    start_variant = list(group['Variant'])[0]
    start_variantidx = list(group['Variant index'])[0]
    start_lifecyle = 'Start'

    group.loc[-1] = [_, start_activity, start_ts, start_variant, start_variantidx, start_lifecyle]
    group.index +=1
    group = group.sort_index()
    concating.append(group)

dfk = pd.concat(concating).sort_values(by='Complete Timestamp')
dfk.to_csv('./data/loan_baseline.pnml_noise_0.0_iteration_1_seed_98070_sample.csv', index=False)