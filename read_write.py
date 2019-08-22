import pickle

def tosave(file_path,data):
    pickle.dump(data, open(file_path, 'wb'),protocol = pickle.HIGHEST_PROTOCOL)
    print('此数据已经成功保存')
def toload(file_path):
    data = pickle.load(open(file_path, 'rb'))
    return data