#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python环境：python 3.7 conda base
"""

import gym
import random
import numpy as np
import time
# import automata
#from policy import CustomEpsGreedyQPolicy
import matplotlib.pyplot as plt
import csv 
import pandas as pd
import seaborn as sns

xlabel_name="Type 1 Time"
# xlabel_name="index"


intel = pd.read_csv('C:/Users/davue/Desktop/automata_gym(1)/automata_gym/automata/envs/dados/approve-A-90-fsInt.csv')
randomic = pd.read_csv('C:/Users/davue/Desktop/automata_gym(1)/automata_gym/automata/envs/dados/approve-A-90-fsRnd.csv')
redo = pd.read_csv('C:/Users/davue/Desktop/automata_gym(1)/automata_gym/automata/envs/dados/approve-A-90-redo.csv')

intel = intel.drop(["Unnamed: 0"],axis=1)
randomic = randomic.drop(["Unnamed: 0"], axis=1)
redo = redo.drop(["Unnamed: 0"], axis=1)





redoRelation = redo.values.tolist()
intel = intel.values.tolist()
randomic = randomic.values.tolist()

a=[]
for i in range(len(redoRelation)):
    for j in range(len(intel)):
        if(redoRelation[i][1]==intel[j][0] and redoRelation[i][3]=='Supervisory+RL'):
            if(redoRelation[i][2]=='redo_A'):
                if(intel[j][1]+intel[j][3]!=0):
                    a.append((redoRelation[i][0]/(intel[j][1]+intel[j][3]), 'Rework Type 1', redoRelation[i][3], redoRelation[i][1])) 
                else:
                    a.append((0, 'Rework Type 1', redoRelation[i][3], redoRelation[i][1])) 
            else:
                if(intel[j][2]+intel[j][4]!=0):
                    a.append((redoRelation[i][0]/(intel[j][2]+intel[j][4]), 'Rework Type 2', redoRelation[i][3], redoRelation[i][1]))
                else:
                    a.append((0, 'Rework Type 2', redoRelation[i][3], redoRelation[i][1])) 
        if(redoRelation[i][3]=='Supervisory' and redoRelation[i][1]==randomic[j][0]):
            if(redoRelation[i][2]=='redo_A'):
               if(randomic[j][1]+randomic[j][3]!=0):
                     a.append((redoRelation[i][0]/(randomic[j][1]+randomic[j][3]), 'Rework Type 1', redoRelation[i][3], redoRelation[i][1])) 
               else:
                     a.append((0, 'Rework Type 1', redoRelation[i][3], redoRelation[i][1]))
            else:
                if(randomic[j][2]+randomic[j][4]!=0):
                    a.append((redoRelation[i][0]/(randomic[j][2]+randomic[j][4]), 'Rework Type 2', redoRelation[i][3], redoRelation[i][1]))
                else:
                    a.append((0, 'Rework Type 2', redoRelation[i][3], redoRelation[i][1]))
                    
a = pd.DataFrame(a, columns=["Reworks/Cars Produced", "Event", "Method", xlabel_name])

sns.lineplot(x=xlabel_name, y="Reworks/Cars Produced", style="Method", hue="Event", data=a, markers=True)
plt.savefig('25.png')
plt.show()
plt.show()


