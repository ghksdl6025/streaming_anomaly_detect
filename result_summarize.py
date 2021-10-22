import pandas as pd
import pickle as pkl

df = pd.DataFrame(columns=['Noise',0.01,0.05,0.1,0.15,0.2,0.25])
df['Noise'] = [0.15,0.125,0.099,0.075,0.0499,0.0249]
df = df.set_index(df['Noise'],drop=True).drop(columns=['Noise'])

# Noise	0.01	0.05	0.1	0.15	0.2	0.25
# 0.15	0.320189274	0.305403289	0.27925117	0.251525604	0.231428571	0.218681798
# 0.125	0.291205591	0.275956284	0.248344371	0.225796076	0.20921472	0.195316658
# 0.099	0.267934313	0.256410256	0.229929001	0.210451977	0.194362642	0.182453416
# 0.075	0.199134199	0.195674044	0.174399337	0.157036512	0.143979881	0.131133672
# 0.0499	0.16828721	0.171855542	0.151937984	0.134941912	0.122924901	0.112182922
# 0.0249	0.086915888	0.08622079	0.079776067	0.070481928	0.062368973	0.056331471


print(df.head)