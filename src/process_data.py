import pickle
import os
from os import listdir
from PIL import Image
import numpy as np
import random
from PIL import Image
path_folder ="./data_food/food-20/images/"
size =(128, 128)
def load_all_folder(folder_parent):

    folder_child = [os.path.join(folder_parent, f) for f in listdir(folder_parent) if os.path.isdir(os.path.join(folder_parent, f))]
    return folder_child
def load_all_path_file(data_folder) :
    image_paths = []
    onlyfiles = [os.path.join(data_folder, f) for f in listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]
    for str_file_path in onlyfiles:
        if str_file_path.endswith(".jpg") or str_file_path.endswith(".JPG"):
            image_paths.append(str_file_path)
    return image_paths
def data_binary(folder_load):
    list_train = np.zeros([1,128, 128,3], dtype="int32")
    label_train = np.array([1], dtype="int32")
    list_test = np.zeros([1, 128, 128, 3])
    label_test = np.array([1], dtype ="int32")
    folders_food = load_all_folder(folder_load)
    i=0

    for folder in folders_food:
        print("folder thu ", i)
        img_paths = load_all_path_file(folder)
        number_img_file = 0
        list_img =  np.zeros([1,128,128,3], dtype="int32")
        label_img_file = np.array([1])
        for path_file in img_paths:
            f = open(path_file, "r")

            img = Image.open(path_file)
            img = img.resize(size, Image.ANTIALIAS)

            arr = np.asarray(img, dtype="int32")
            # print(arr.shape)
            # assert False
            arr= np.array([arr])
            print("arr",arr.shape)
            print("list_img", list_img.shape)
            l = np.array([int(i)])
            list_img = np.concatenate((list_img, arr), axis=0)
            label_img_file = np.concatenate((label_img_file, l), axis =0)
            # print(list_train)
            number_img_file  +=1
            print("      file thu ", number_img_file)
        i+=1

        list_img = np.delete(list_img, 0,0)
        label_img_file = np.delete(label_img_file, 0,0)
        if (i==0):
            list_train = np.delete(list_train, 0, 0)
            label_train = np.delete(label_train, 0,0)
            list_test = np.delete(list_test, 0,0)
            label_test = np.delete(label_test, 0,0)


        number_train = int(0.8*number_img_file )
        list_train = np.concatenate((list_train,list_img[:number_train, :, :,:] ), axis=0)
        label_train = np.concatenate((label_train, label_img_file[:number_train]), axis=0)
        list_test = np.concatenate((list_test, list_img[number_train:, :, :, :]), axis =0)
        label_test = np.concatenate((label_test, label_img_file[number_train:]), axis =0)
       # images
       #  print("train", list_train)
       #  print("label_train", label_train)
       #  print("test", list_test)
       #  print("label", label_test)
    return list_train,label_train, list_test, label_test
def save_image(list_train, label_train, list_test, label_test):
    data_train ={"X": list_train, "Y":label_train}
    data_test ={"X": list_test, "Y" : label_test}
    with open("./processed_data_128/train.pickle", "wb") as f:

        pickle.dump(data_train, f)
    with open ("./processed_data_128/test.pickle", "wb") as f:
        pickle.dump(data_test, f)


list_train , label_train, list_test, label_test = data_binary(path_folder)
save_image(list_train, label_train, list_test, label_test)

# img = Image.open("/home/phuongnm/PycharmProjects/ResNetv2_CIFAR-master/frame1.jpg")
#
# img = img.resize(size, Image.ANTIALIAS)
# print(img)
# assert False
# # arr = np.asarray(img, dtype="int32")
# arr= np.array([arr])
# a = np.array([[[0,2,3],[1,2,4],[1,5,6]],[[1,2,3],[1,2,4],[1,5,6]]])
# print(np.delete(a, 0,0))
# print(int(3/2))
# a =np.array([1])
# l = np.array([2])
# a = np.delete(a, 0,0)
# a = np.concatenate((a,l), axis = 0)
# print(a)
