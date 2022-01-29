import pickle
import numpy as np
import cv2
import os
data_type = 'Imaged'

X = pickle.load( open( "./dataset/processed_data/"+data_type+"/X.p", "rb" ) )
y = pickle.load( open( "./dataset/processed_data/"+data_type+"/y.p", "rb" ) )
p = pickle.load( open( "./dataset/processed_data/"+data_type+"/p.p", "rb" ) )
dim = pickle.load( open( "./dataset/processed_data/"+data_type+"/dim.p", "rb" ) )

print('X', X.shape,'y', y.shape,'p',len(set([pi[0] for pi in p])))

X = X[:5890]
y = y[:5890]
p = p[:5890]
print('X', X.shape,'y', y.shape,'p',len(set([pi[0] for pi in p])))

def convert_mesh(X):
    
    mesh = np.zeros((X.shape[0], X.shape[2], 10, 11, 1))
    X = np.swapaxes(X, 1, 2)
    
    # 1st line
    mesh[:, :, 0, 4:7, 0] = X[:,:,21:24]; print('1st finished')
    
    # 2nd line
    mesh[:, :, 1, 3:8, 0] = X[:,:,24:29]; print('2nd finished')
    
    # 3rd line
    mesh[:, :, 2, 1:10, 0] = X[:,:,29:38]; print('3rd finished')
    
    # 4th line
    mesh[:, :, 3, 1:10, 0] = np.concatenate((X[:,:,38].reshape(-1, X.shape[1], 1),\
                                          X[:,:,0:7], X[:,:,39].reshape(-1, X.shape[1], 1)), axis=2)
    print('4th finished')
    
    # 5th line
    mesh[:, :, 4, 0:11, 0] = np.concatenate((X[:,:,(42, 40)],\
                                        X[:,:,7:14], X[:,:,(41, 43)]), axis=2)
    print('5th finished')
    
    # 6th line
    mesh[:, :, 5, 1:10, 0] = np.concatenate((X[:,:,44].reshape(-1, X.shape[1], 1),\
                                        X[:,:,14:21], X[:,:,45].reshape(-1, X.shape[1], 1)), axis=2)
    print('6th finished')
               
    # 7th line
    mesh[:, :, 6, 1:10, 0] = X[:,:,46:55]; print('7th finished')
    
    # 8th line
    mesh[:, :, 7, 3:8, 0] = X[:,:,55:60]; print('8th finished')
    
    # 9th line
    mesh[:, :, 8, 4:7, 0] = X[:,:,60:63]; print('9th finished')
    
    # 10th line
    mesh[:, :, 9, 5, 0] = X[:,:,63]; print('10th finished')
    
    return mesh


def split_data(X, y, p, test_ratio, set_seed, user_independent):
    # Shuffle trials
    np.random.seed(set_seed)
    if user_independent == 'user_independent':
        trials = len(set([ele[0] for ele in p]))
    else:    
        trials = X.shape[0]
    print('trial', trials)
    shuffle_indices = np.random.permutation(trials)
    
    print('-- shaffleing X, y, p')
    if user_independent == 'user_independent':
        # Create a dict with empty list as default value.
        d = defaultdict(list)
        # print(y.shape, np.array([ele[0] for ele in y]).shape)
        for index, e in enumerate([ele[0] for ele in y]):
            # print('index', index, 'e', e)
            d[e].append(index)
        new_indexes = []
        for i in shuffle_indices:
            new_indexes += d[i]
        X = X[new_indexes]
        y = y[new_indexes]
        p = p[new_indexes]
        train_size = 0
        for i in shuffle_indices[:int(trials*(1-test_ratio))]:
            train_size += len(d[i])
                
    else:
        X = X[shuffle_indices]
        y = y[shuffle_indices]
        p = p[shuffle_indices]
        # Test set seperation
        train_size = int(trials*(1-test_ratio)) 
    
    print('-- split X, y, p in train-test',train_size)
    
    # X_train, X_test, y_train, y_test, p_train, p_test
    return  X[:train_size,:,:], X[train_size:,:,:], y[:train_size,:], y[train_size:,:], p[:train_size,:], p[train_size:,:]
               
def prepare_data(X, y, p, test_ratio, return_mesh, set_seed, user_independent):
    
    # y encoding
    # oh = OneHotEncoder(categories='auto')
    # y = oh.fit_transform(y).toarray()
    
    print('Split dataset:')
    #X_train, X_test, y_train, y_test, p_train, p_test = split_data(X, y, p, test_ratio, set_seed, user_independent)
                                    
    # Z-score Normalization
    def scale_data(X):
        shape = X.shape
        for i in range(shape[0]):
            # Standardize a dataset along any axis
            # Center to the mean and component wise scale to unit variance.
            X[i,:, :] = scale(X[i,:, :])
            if i%int(shape[0]//10) == 0:
                print('{:.0%} done'.format((i+1)/shape[0]))   
        return X
    
    print('Scaling data')
    print('-- X train-test along any axis')
    X  = scale_data(X)# , scale_data(X_test)
    
    if return_mesh:
        print('Creating mesh')
        print('-- X train-test to mesh')
        X = convert_mesh(X)#, convert_mesh(X_test)
    
    return X, y, p

test_rate = 0
seed = 42
split_type = 'user_dependent'
from sklearn.preprocessing import scale

Xs, ys, ps = prepare_data(X, y, p, test_rate, True, seed, split_type)

print('\n\n\nX_train', Xs.shape, 
    'y_train', ys.shape, 
    'p_train', len(set([p[0] for p in ps])))


Xs = Xs[::2]
print(Xs.shape)
ys = ys[::2]
print(ys.shape)
ps = ps[::2]
print(ps.shape)

size = (Xs.shape[0]*Xs.shape[1], Xs.shape[2], Xs.shape[3], Xs.shape[4]) 
sX = np.reshape(np.ravel(Xs), size).squeeze()
print(sX.shape)


sy = []
for y in ys:
    # print([y for i in range(10)])
    sy.extend([y for i in range(10)])
print(np.array(sy).shape)


def save_img(X, y, base_dir='/imgs/', person_folder='S001/', scale = 1000, want_scale = False):
    USERDIR = base_dir + person_folder
    for ty in [ 'f'+str(k)+'/' for k in set([yy[0] for yy in y])]:
        if not os.path.exists(os.path.dirname(USERDIR + ty)):
            os.makedirs(os.path.dirname(USERDIR + ty))
            print('done '+str(ty) )
    maxi = X.max()
    mini = X.min()
    size = (10,11)
    if want_scale:
        size = tuple([scale*x for x in (10,11)])
        
    # out = cv2.VideoWriter(USERDIR + 'project.avi',cv2.VideoWriter_fourcc(*'MJPG'),2, size )
    
    if want_scale:
        e0 = np.ones(scale)
        M0 = np.zeros((11, scale*11 ))
        for i in range(M0.shape[0]):
            M0[i][i*scale:(i+1)*scale] = e0
            # print(M0[i][i*scale:(i+1)*scale])

        e1 = np.ones(scale)
        M1 = np.zeros((10, scale*10 ))
        for i in range(M1.shape[0]):
            M1[i][i*scale:(i+1)*scale] = e1
            # print(M1[i][i*scale:(i+1)*scale])
    
    d = {}
    for k in  set([yy[0] for yy in y]):
        d[k] = []
    
    imgs = []
    o_ty = 0
    # print([yy for yy in y])
    for i, (ele, ty) in enumerate(zip(X, [yy[0] for yy in y] )):
        #print(ele.shape)
        #if not (o_ty == ty):
        #    d[o_ty].append(-1)
        d[ty].append(i)
        #o_ty = ty
        if want_scale: img = ((((ele.squeeze()).dot(M0)).T).dot(M1)).T
        else: img = ele.squeeze()
        print('done', i)
        # ele = cv2.resize(ele, size, cv2.INTER_AREA)
        img = cv2.applyColorMap((((np.array(img)-mini)/(maxi-mini))*255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
        imgs.append(img)
        # out.write(img)
        # if i%100 == 0:
        # cv2.imwrite(USERDIR + 'f'+str(ty)+ '/img_'+str(i)+'.jpg', np.array(img))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    # out.release()
    return d, imgs

want_scale = False
dic, imgs = save_img(sX, sy, './imgs/', 'S001/')

from operator import itemgetter
from itertools import *

size = (11,10)
if want_scale:
    size = tuple([scale*x for x in size])
    
for ty in set([yy[0] for yy in sy]):
    lis = []
    for k, g in groupby(enumerate(dic[ty]), lambda kv :kv[0]-kv[1] ):
        lis.append(map(itemgetter(1), g)) 
    
    for i, a in enumerate(lis):
        print(ty, i)
        # out = cv2.VideoWriter(USERDIR + 'project.avi',cv2.VideoWriter_fourcc(*'MJPG'),2, size )
        out = cv2.VideoWriter('./imgs/'+'S001/f'+str(ty)+'/class_'+str(ty)+'_'+str(i)+'.avi',cv2.VideoWriter_fourcc(*'MJPG'),2, size )
        for j in [sX[k] for k in a]:
            out.write(j)
        out.release()