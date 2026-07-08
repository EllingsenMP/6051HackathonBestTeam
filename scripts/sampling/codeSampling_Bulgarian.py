import pandas as pd
import numpy as np

data = pd.read_excel('../../data/eduBotTranslated - Bulgarian.xlsx')

sample_100 = data.sample(100, random_state = 6051)
sample_100.to_excel('../data/eduBotSampleBulgarian100.xlsx')

sample_30 = data.sample(30, random_state = 6051)
sample_30.to_excel('../data/eduBotSampleBulgarian30.xlsx')
