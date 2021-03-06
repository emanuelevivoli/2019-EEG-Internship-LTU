import re
import requests
import pathlib
import urllib
import os

#%%
CONTEXT = 'pn4/'
MATERIAL = 'eegmmidb/'
URL = 'https://www.physionet.org/' + CONTEXT + MATERIAL

USERDIR = './data/' # Change this directory according to your setting

page = requests.get(URL).text
FOLDERS = sorted(list(set(re.findall(r'S[0-9]+', page))))

FOLDERS = [     'S014', 'S034', 'S037', 'S041', 'S051', 'S064', 'S069', 'S072', 'S073', 'S074', 'S076', 
                'S088', 'S092', 'S100', 'S102', 'S106', 'S109'] 
print(FOLDERS[0])

URLS = [URL+x+'/' for x in FOLDERS]

# # Warning: Executing this block will create folders
for folder in FOLDERS:
    pathlib.Path(USERDIR +'/'+ folder).mkdir(parents=True, exist_ok=True)
    


 # Warning: Executing this block will start downloading data
for i, folder in enumerate(FOLDERS):
    page = requests.get(URLS[i]).text
    subs = list(set(re.findall(r'S[0-9]+R[0-9]+', page)))
    
    print('Working on {}, {:.1%} completed'.format(folder, (i+1)/len(FOLDERS)))
    for sub in subs:
        urllib.request.urlretrieve(URLS[i]+sub+'.edf', os.path.join(USERDIR, folder, sub+'.edf'))
        urllib.request.urlretrieve(URLS[i]+sub+'.edf.event', os.path.join(USERDIR, folder, sub+'.edf.event'))