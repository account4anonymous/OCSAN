import os
import random
import cv2
from PIL import Image
import tensorflow as tf
import numpy as np
from tqdm import tqdm

class Parameters:
    batch_size = 8
    epoch = 100
    in_width = 128
    in_height = 128
    in_channel = 3
    in_frame = 10
    learning_rate = 0.0001


class train_same_people_DataSequence(tf.keras.utils.Sequence):
    def __init__(self, path: str, batch_size: int,number:str,famous:bool):
        assert batch_size % 2 == 0
        self.batch_size = batch_size
        self.video_folder_list = list()
        self.label_list = list()
        self.real_path = path
        self.num = number
        if famous:
            self.video_folder_list,self.same_people_list = self.famous_single_train_X(self.real_path,self.num)
        else:
            self.video_folder_list,self.same_people_list = self.single_train_X(self.real_path,self.num)
        #
        

    def __len__(self):
        return int(np.floor(len(self.video_folder_list) / float(self.batch_size)))


    def __getitem__(self, batch_idx):
        batch_file = self.video_folder_list[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
        same_people_file = self.same_people_list[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
        x_batch_face = np.asarray([np.array(Image.open(frame)) for frame in same_people_file]).astype('float32')/127.5 -1
        x_batch_face_mini = np.asarray([np.array(Image.open(frame).resize((32,32))) for frame in same_people_file]).astype('float32')/127.5 -1

        y_batch_face = np.asarray([np.array(Image.open(frame)) for frame in batch_file]).astype('float32')/127.5 -1
        y_batch_face_mini = np.asarray([np.array(Image.open(frame).resize((32,32))) for frame in batch_file]).astype('float32')/127.5 -1

        return [x_batch_face_mini,x_batch_face],[y_batch_face_mini,y_batch_face]

    def single_train_X(self,real_path,person):
        video_folder_list = list()
        same_people_list = list()
        person_folder_name = list()
        for folder in os.listdir(real_path):
            if person in folder: #if person in folder:
                person_folder_name.append(folder)
        for folder_name in person_folder_name:
            if os.path.isdir(os.path.join(real_path, folder_name)):
                frame_name_list = sorted([os.path.join(real_path, folder_name, name)
                                          for name in os.listdir(os.path.join(real_path, folder_name))
                                          if (os.path.splitext(name)[-1] == '.bmp' or os.path.splitext(name)[-1] == '.png')])

                temp_00 = list()
                temp_01 = list()
                temp_02 = list()
                for frame in frame_name_list:
                    if '_00_' in frame:
                        temp_00.append(frame)
                    elif '_01_' in frame:
                        temp_01.append(frame)
                    elif '_02_' in frame:
                        temp_02.append(frame)
                if len(temp_00) > len(temp_01):
                    max_num = len(temp_00)
                    real_video_folder_list = temp_00
                    if len(temp_00) > len(temp_02):
                        max_num = len(temp_00)
                        real_video_folder_list = temp_00
                    else:
                        max_num = len(temp_02)
                        real_video_folder_list = temp_02
                else:
                    max_num = len(temp_01)
                    real_video_folder_list = temp_01
                    if len(temp_01) > len(temp_02):
                        max_num = len(temp_01)
                        real_video_folder_list = temp_01
                    else:
                        max_num = len(temp_02)
                        real_video_folder_list = temp_02
            
                #real_video_folder_list = frame_name_list
                video_folder_list.extend(real_video_folder_list)
                #print(real_video_folder_list[0])
                random.shuffle(frame_name_list)
                #print(frame_name_list[0])

                same_people_list.extend(frame_name_list)
                random.shuffle(same_people_list)
                
        name_list = list(zip(video_folder_list,same_people_list))
        random.shuffle(name_list)
        video_folder_list,same_people_list = list(zip(*name_list))
        #list_len = (len(video_folder_list) // self.batch_size) * self.batch_size

        print(len(video_folder_list))
        return video_folder_list,same_people_list
        
    def famous_single_train_X(self,real_path,person):
        video_folder_list = list()
        same_people_list = list()
        person_folder_name = list()
        for folder in os.listdir(real_path):
            if person in folder: #if person in folder:
                person_folder_name.append(folder)
        for folder_name in person_folder_name:
            if os.path.isdir(os.path.join(real_path, folder_name)):
                frame_name_list = sorted([os.path.join(real_path, folder_name, name)
                                          for name in os.listdir(os.path.join(real_path, folder_name))
                                          if (os.path.splitext(name)[-1] == '.bmp' or os.path.splitext(name)[-1] == '.png')])
                    
            
            
                real_video_folder_list = frame_name_list
                video_folder_list.extend(real_video_folder_list)
                #print(real_video_folder_list[0])
                random.shuffle(frame_name_list)
                #print(frame_name_list[0])

                same_people_list.extend(frame_name_list)
                random.shuffle(same_people_list)
                    
        name_list = list(zip(video_folder_list,same_people_list))
        random.shuffle(name_list)
        video_folder_list,same_people_list = list(zip(*name_list))
            
        print(len(video_folder_list))    
        return video_folder_list,same_people_list
        


