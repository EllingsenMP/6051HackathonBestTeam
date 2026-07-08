import pandas as pd
import numpy as np

data = pd.read_csv('../data/chatbot_tutor_50k_dataset.csv')

sample_5000 = data.sample(5000, random_state = 6051)
sample_5000.to_csv('../data/eduBotSample5000.csv')

sample_100 = data.sample(100, random_state = 6051)
sample_100.to_csv('../data/eduBotSample100.csv')

sample_10 = data.sample(10, random_state = 6051)
sample_10.to_csv('../data/eduBotSample10.csv')

sample_3 = data.sample(3, random_state = 6051)
sample_3.to_csv('../data/eduBotSample3.csv')

sample_30 = data.sample(30, random_state = 6051)
sample_30.to_csv('../data/eduBotSample30.csv')
