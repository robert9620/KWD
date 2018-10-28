import numpy as np
import pandas as pd

from KNN import KNN

listaUczaca = np.array(pd.read_csv("data-learning.csv", header=None))
listaTestujaca = np.array(pd.read_csv("data-test.csv", header=None))

listaTestujacaBezEtykiet = np.delete(listaTestujaca, len(listaTestujaca[0])-1, axis=1)
etykietyListyTestujacej = listaTestujaca[:,len(listaTestujaca[0])-1]

ai = KNN(3, listaUczaca)
print(ai.predict(listaTestujacaBezEtykiet))
print(str(ai.score(listaTestujacaBezEtykiet, etykietyListyTestujacej))+"% poprawnych wyników")
