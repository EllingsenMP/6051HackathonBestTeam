import pandas as pd
import numpy as np

data = pd.read_csv('../data/eduBotTranslated - Bulgarian.csv')

sample_100 = data.sample(100, random_state = 6051)
sample_100.to_csv('../data/eduBotSampleBulgarian100.csv')

sample_30 = data.sample(30, random_state = 6051)
sample_30.to_csv('../data/eduBotSampleBulgarian30.csv')
