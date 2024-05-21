import zipfile
from urllib.request import urlretrieve
import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def download_uci_dataset():
    os.makedirs('zips', exist_ok=True)
    url = 'https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip'
    urlretrieve(url, 'zips/UCIHAR.zip')

def extract_uci_dataset():
    os.makedirs('data/UCIHAR', exist_ok=True)
    with zipfile.ZipFile('zips/UCIHAR.zip', 'r') as zip_ref:
        zip_ref.extractall('data/UCIHAR')
    os.makedirs('data/UCIHAR/dataset', exist_ok=True)
    with zipfile.ZipFile('data/UCIHAR/UCI HAR Dataset.zip', 'r') as zip_ref:
        zip_ref.extractall('data/UCIHAR/dataset')

def read_files(validation=True):
    np.random.seed(42)
    base_dir = 'data/UCIHAR/dataset/UCI HAR Dataset'
    # Training data
    # Train users
    train_users = pd.read_csv(f'{base_dir}/train/subject_train.txt', header=None)
    # Train accelerometer data
    train_acc_x = pd.read_csv(f'{base_dir}/train/Inertial Signals/body_acc_x_train.txt', delim_whitespace=True, header=None)
    train_acc_y = pd.read_csv(f'{base_dir}/train/Inertial Signals/body_acc_y_train.txt', delim_whitespace=True, header=None)
    train_acc_z = pd.read_csv(f'{base_dir}/train/Inertial Signals/body_acc_z_train.txt', delim_whitespace=True, header=None)
    # Train gyroscope data
    train_gyro_x = pd.read_csv(f'{base_dir}/train/Inertial Signals/body_gyro_x_train.txt', delim_whitespace=True, header=None)
    train_gyro_y = pd.read_csv(f'{base_dir}/train/Inertial Signals/body_gyro_y_train.txt', delim_whitespace=True, header=None)
    train_gyro_z = pd.read_csv(f'{base_dir}/train/Inertial Signals/body_gyro_z_train.txt', delim_whitespace=True, header=None)
    # Train total acc data
    train_total_acc_x = pd.read_csv(f'{base_dir}/train/Inertial Signals/total_acc_x_train.txt', delim_whitespace=True, header=None)
    train_total_acc_y = pd.read_csv(f'{base_dir}/train/Inertial Signals/total_acc_y_train.txt', delim_whitespace=True, header=None)
    train_total_acc_z = pd.read_csv(f'{base_dir}/train/Inertial Signals/total_acc_z_train.txt', delim_whitespace=True, header=None)

    # Train labels
    train_y = pd.read_csv(f'{base_dir}/train/y_train.txt', header=None)

    # Test data
    # Test users
    test_users = pd.read_csv(f'{base_dir}/test/subject_test.txt', header=None)
    # Test accelerometer data
    test_acc_x = pd.read_csv(f'{base_dir}/test/Inertial Signals/body_acc_x_test.txt', delim_whitespace=True, header=None)
    test_acc_y = pd.read_csv(f'{base_dir}/test/Inertial Signals/body_acc_y_test.txt', delim_whitespace=True, header=None)
    test_acc_z = pd.read_csv(f'{base_dir}/test/Inertial Signals/body_acc_z_test.txt', delim_whitespace=True, header=None)
    # Test gyroscope data
    test_gyro_x = pd.read_csv(f'{base_dir}/test/Inertial Signals/body_gyro_x_test.txt', delim_whitespace=True, header=None)
    test_gyro_y = pd.read_csv(f'{base_dir}/test/Inertial Signals/body_gyro_y_test.txt', delim_whitespace=True, header=None)
    test_gyro_z = pd.read_csv(f'{base_dir}/test/Inertial Signals/body_gyro_z_test.txt', delim_whitespace=True, header=None)
    # Test total acc data
    test_total_acc_x = pd.read_csv(f'{base_dir}/test/Inertial Signals/total_acc_x_test.txt', delim_whitespace=True, header=None)
    test_total_acc_y = pd.read_csv(f'{base_dir}/test/Inertial Signals/total_acc_y_test.txt', delim_whitespace=True, header=None)
    test_total_acc_z = pd.read_csv(f'{base_dir}/test/Inertial Signals/total_acc_z_test.txt', delim_whitespace=True, header=None)

    # Test labels
    test_y = pd.read_csv(f'{base_dir}/test/y_test.txt', header=None)

    train_data = pd.concat([train_users, train_gyro_x, train_gyro_y, train_gyro_z, train_total_acc_x, train_total_acc_y, train_total_acc_z, train_acc_x, train_acc_y, train_acc_z], axis=1)
    test_data = pd.concat([test_users, test_gyro_x, test_gyro_y, test_gyro_z, test_total_acc_x, test_total_acc_y, test_total_acc_z, test_acc_x, test_acc_y, test_acc_z], axis=1)
    
    if validation:
        # Choosing the users for training and validation
        users_for_train = np.random.choice(train_users.iloc[:,0].unique(), 7, replace=False)
        users_for_validation = np.setdiff1d(train_users.iloc[:,0].unique(), users_for_train)
        # print(users_for_train, users_for_validation)
        validation_y = train_y[train_data.iloc[:,0].isin(users_for_validation)]
        train_y = train_y[train_data.iloc[:,0].isin(users_for_train)]
        
        validation_data = train_data[train_data.iloc[:,0].isin(users_for_validation)]
        train_data = train_data[train_data.iloc[:,0].isin(users_for_train)]
    
        train_data = train_data.iloc[:,1:]
        validation_data = validation_data.iloc[:,1:]
        test_data = test_data.iloc[:,1:]
    else:
        train_data = train_data.iloc[:,1:]
        test_data = test_data.iloc[:,1:]
        validation_data = None
        validation_y = None
    return train_data, train_y, validation_data, validation_y, test_data, test_y




def timeserie2image(data, filename=None, indexes=[1,2,3,4,5,6,7,8,9,1,3,5,7,9,2,4,6,8,1,4,7,1,5,8,2,5,9,3,6,9,4,8,3,7,2,6]):
    if data.shape[1] == 9:
        data = data.T
    data = [data[i-1, :] for i in indexes]
    data = np.fft.fft2(data)
    data = np.fft.fftshift(data)
    # Save the image in jpeg format
    data = np.abs(data)
    # data = data / np.max(data)
    # data = (100 + data * 155).astype(np.uint8)
    # im = Image.fromarray(data)
    # im.save(filename)
    # Turn all over 255 to 255
    data[data > 255] = 255
    data[data < 0] = 0
    if filename:
        plt.imshow(data)
        plt.colorbar()
        plt.show()
        plt.close()
        print(data.shape)
    return data