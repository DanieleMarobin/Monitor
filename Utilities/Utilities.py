import os
import shutil
import pickle
from datetime import datetime as dt

def log(message):
    print(dt.now().strftime("%H:%M:%S"),'-' + message)

def updated_today(file):
    if not os.path.exists(file): return False
        
    (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = os.stat(file)
    modified = dt.date(dt.fromtimestamp(mtime))
    today = dt.date(dt.today())
    return (modified == today)

def serialize(obj, file_path, comment=True):
    if comment: print('Start Serializing:', file_path)
    pickle.dump(obj, open(file_path, "wb" ))
    if comment: print('Done Serializing:', file_path)

def deserialize(file_path, comment=True):
    if comment:print('Start Deserializing:', file_path)
    obj= pickle.load(open(file_path, "rb"))
    if comment:print('Done Deserializing:', file_path)
    return obj

def show_excel_index(df,file='check', index=True):
    df.to_csv(file+'.csv', index=index,)
    program =r'"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE"'
    os.system("start " +program+ " "+file+'.csv')

def show_excel(df,file='check', index=False):
    df.to_csv(file+'.csv', index=index,)
    program =r'"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE"'
    os.system("start " +program+ " "+file+'.csv')

def open_cur_folder():
    os.system('start explorer ' + os.getcwd())


def copy_folder(source_folder,destination_folder,verbose=False):
    # fetch all files
    for file_name in os.listdir(source_folder):
        # construct full file path
        source = source_folder + file_name
        destination = destination_folder + file_name
        # copy only files
        if os.path.isfile(source):
            shutil.copy(source, destination)
            if verbose: print('copied', file_name)
    print('-------- All Copied --------')