import h5py
import numpy as np
def readH5file(dataSet='awa1',type='binary'):
    filename = "./datasets/"+dataSet+"/data.h5"
    file = h5py.File(filename, 'r')
    #==========================================#
    train_x = np.array(file['train']['X'])
    train_a = np.array(file['train']['A'][type])
    train_y = np.array(file['train']['Y'])
    # ==========================================#
    val_x = np.array(file['val']['X'])
    val_y = np.array(file['val']['Y'])
    val_a = np.array(file['val']['A'][type])
    # ==========================================#
    test_x = np.array(file['test']['unseen']['X'])
    test_a = np.array(file['test']['unseen']['A'][type])
    test_y = np.array(file['test']['unseen']['Y'])
    return (train_x, train_y, train_a), (test_x, test_y,test_a), (val_x, val_y,val_a)
def readH5file2(dataSet='awa1',type='binary'):
    filename = "./datasets/"+dataSet+"/data.h5"
    file = h5py.File(filename, 'r')
    #==========================================#
    train_x = np.array(file['train']['X'])
    train_a = np.array(file['train']['A'][type])
    train_y = np.array(file['train']['Y'])
    # ==========================================#
    test_x = np.array(file['test']['seen']['X'])
    test_a = np.array(file['test']['seen']['A'][type])
    test_y = np.array(file['test']['seen']['Y'])
    return (train_x, train_y, train_a), (test_x, test_y,test_a)
def readForDap(dataSet='awa1',type='binary'):
    filename = "./datasets/"+dataSet+"/data.h5"
    file = h5py.File(filename, 'r')
    #==========================================#
    train_x = np.array(file['train']['X'])
    train_a = np.array(file['train']['A'][type])
    train_y = np.array(file['train']['Y'])

    # ==========================================#
    test_xs = np.array(file['test']['seen']['X'])
    test_as = np.array(file['test']['seen']['A'][type])
    test_ys = np.array(file['test']['seen']['Y'])
    test_x = np.array(file['test']['unseen']['X'])
    test_a = np.array(file['test']['unseen']['A'][type])
    test_y = np.array(file['test']['unseen']['Y'])
    from keras.models import load_model
    generator = load_model('./model/generator.h5', compile=False)
    noise   = np.random.normal(0, 1, (5685, 85))
    temp_x  = generator.predict([noise, test_a])

    idx = np.random.randint(0, temp_x.shape[0], 300)
    features, labels, attr = temp_x[idx], test_y[idx], test_a[idx]

    test_x = np.concatenate((test_x, test_xs), axis=0)
    test_a = np.concatenate((test_a, test_as), axis=0)
    test_y = np.concatenate((test_y, test_ys), axis=0)

    final_x = np.concatenate((train_x, features), axis=0)
    final_a = np.concatenate((train_a, attr), axis=0)
    final_y = np.concatenate((train_y, labels), axis=0)
    return (final_a,final_x,final_y), (test_a,test_x, test_y)

def numberOfClass(dataSet='awa1'):
    filename = "./datasets/"+dataSet+"/data.h5"
    file = h5py.File(filename, 'r')
    return np.unique(file['train']["Y"]).shape[0]+np.unique(file['test']['unseen']['Y']).shape[0]
