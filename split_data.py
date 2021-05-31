"""Splitting data for training & testing"""

import csv
import argparse
import random
from typing import Iterator, List

data: List[dict]= []

with open("data/clickbait_data.csv", "r") as source:
        for row in csv.DictReader(source, delimiter=","):
            data.append({row["headline"]:row["clickbait"]})

x = random.randint(0,200)
random.seed(x)
random.shuffle(data)

size = len(data)

train = data[:8*size//10]
test = data[8*size//10:]

for article in train:
    with open("train","a") as sink:
        print(article, file = sink)

for article in test:
    with open("test","a") as sink:
        print(article, file = sink)