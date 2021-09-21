import pandas as pd
from river import tree
import datetime

df = pd.read_csv('./bpic17_sampling2.csv')

df = df.loc[:,['Case ID', 'Activity', 'Resource', 'Start Timestamp']]
df = df.rename({'Case ID':'caseid','Activity':'activity','Resource':'resource', 'Start Timestamp': 'ts'}, axis=1)
df = df.sort_values(by='ts')

groups = df.groupby('caseid')

concating = []
for _, group in groups:
    group = group.reset_index(drop=True)
    finish_tag = 'Finish'
    group_len = len(group['activity'])
    group.loc[group_len-1, 'progress'] = finish_tag
    concating.append(group)

pd_concat = pd.concat(concating)
pd_concat.sort_values(by='ts')
print(pd_concat.head)
df = pd_concat

df['ts'] =  pd.to_datetime(df['ts'])


tslist = []
for pos, act in enumerate(list(df['activity'])):
    if pos % 100 ==0:
        print(pos, "/", len(df['activity']))
    updatets = list(df['ts'])[pos]
    if act == 'End':
        updatets += datetime.timedelta(0,1)
    tslist.append(updatets)

df['ts'] = tslist
df = df.sort_values(by='ts')
groups = df.groupby('caseid')

for _, group in groups:
    group = group.reset_index(drop=True)
    actlist = list(group['activity'])
    endidk = actlist.index('End')

    if endidk +1 != len(actlist):
        print(group)
df = df.drop(['progress'], axis=1)
df.to_csv('./bpic17_sampling2.csv',index=False)