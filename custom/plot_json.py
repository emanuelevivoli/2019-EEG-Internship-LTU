import json as js
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

FOLDER = './jsons/'
JSON_FOLDER = FOLDER + 'files/'
IMGS_FOLDER = FOLDER + 'imgs/'
jsons = sorted(os.listdir(JSON_FOLDER))

# change this
# jsons = jsons[0]
d = {}
d = defaultdict(lambda: -1, d)
epochnum = []
for json in jsons:
    # print(json, json.split('.'))
    name, typ = json.split('.')
    name, plotyp = name.split('-tag-')
    run = name.split('_run')[1][0]
    plotyp = plotyp.replace('epoch_', '')

    with open(JSON_FOLDER + json, 'r') as f:
        data = js.load(f)

    data = np.array(data)
    timewall, epochnum, value = data[:,0], data[:,1], data[:,2]

    print(name, 'run:', run, 'plot type:', plotyp)
    if isinstance(d[run], int) :
        d[run] = {}
    d[run][plotyp] = value

colors = ['r', 'g', 'b']

for key in d.keys():
    subname = []
    for i, sub_key in enumerate(d[key].keys()):
        subname.append(sub_key)
        plt.plot(epochnum, d[key][sub_key], color=colors[int(i%2)])
        if i%2 > 0:
            plt.savefig(IMGS_FOLDER + 'run_'+str(key)+'_'+str(subname)+'.jpg')
            plt.cla()
            subname = []