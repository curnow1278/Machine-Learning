import numpy as np
import pandas as pd
r = 0.5
E = 5.0
n = 10000
I = np.random.randint(0, 10, n)
dataset = {
    'Current_Size': I,
    'Voltage': (-r + np.random.uniform(0, 0.001, n)) * I + E + np.random.uniform(0, 0.01, n)
}

df = pd.DataFrame(dataset)
df.to_csv('train.csv', index=False)