import pandas as pd
import numpy as np

data = pd.read_csv('../../data/eduBotTranslated - Spanish.csv')

sample_100 = data.sample(100, random_state = 6051)
sample_100.to_csv('../data/eduBotSampleSpanish100.csv')

sample_30 = data.sample(30, random_state = 6051)
sample_30.to_csv('../data/eduBotSampleSpanish30.csv')
