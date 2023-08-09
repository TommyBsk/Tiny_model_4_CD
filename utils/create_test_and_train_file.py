import os
import csv
from random import shuffle
from math import floor
from matplotlib.pyplot import imread
       
def get_file_list_from_csv(datapath: str):
    file = open(datapath, "r", encoding='utf-8-sig')
    data_list = list(csv.reader(file))
    file.close()
    data_list_aorta = [data for data in data_list if imread("/home/ramat/data/images/totalSegmentor_abdomen_data/seg/" + data[0].replace("data","seg")).max() > 0]
    return data_list_aorta

def randomize_files(file_list):
    shuffle(file_list)

def get_training_and_testing_sets(file_list):
    split = 0.8
    split_index = floor(len(file_list) * split)
    training = file_list[:split_index]
    testing = file_list[split_index:]
    return training, testing

def create_txt_files_from_lists(file_list: list, file_name: str):
    file = open(file_name, 'w+')
    for file_name in file_list:
        file.write(str(file_name[0]) + "\n")
    file.close()
    
def run():
    file_list = get_file_list_from_csv("/home/ramat/code/Tiny_CD/Tiny_model_4_CD/data/DataFileName_totalSegmentorAbdomen.csv")
    randomize_files(file_list = file_list)
    training_file_list, testing_file_list = get_training_and_testing_sets(file_list=file_list)
    create_txt_files_from_lists(training_file_list, "/home/ramat/code/Tiny_CD/Tiny_model_4_CD/data/train_totalSegmentor_abdomen.txt")
    create_txt_files_from_lists(testing_file_list, "/home/ramat/code/Tiny_CD/Tiny_model_4_CD/data/val_totalSegmentor_abdomen.txt")

if __name__ == "__main__":
    run() 