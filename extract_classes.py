#!/usr/bin/env python
import pandas as pd

df = pd.read_csv('labels.csv')
classes = df['breed']
classes = sorted(set(classes))

print(classes)
