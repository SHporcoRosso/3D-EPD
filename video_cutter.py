import pathlib
import os
import importlib
import sys
import cv2
import numpy as np
import torch
from torch import nn
from skimage.measure import label, regionprops
import PyQt5
from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
file_abspath = os.path.abspath(__file__)
folder_abspath = os.path.dirname(file_abspath)

form_class = uic.loadUiType(os.path.join(folder_abspath, "simple_video.ui"))[0]

def letter_box_resize(img, dsize):

    original_height, original_width = img.shape[:2]
    target_width, target_height = dsize

    ratio = min(
        float(target_width) / original_width,
        float(target_height) / original_height)
    resized_height, resized_width = [
        round(original_height * ratio),
        round(original_width * ratio)
    ]

    img = cv2.resize(img, dsize=(resized_width, resized_height))

    pad_left = (target_width - resized_width) // 2
    pad_right = target_width - resized_width - pad_left
    pad_top = (target_height - resized_height) // 2
    pad_bottom = target_height - resized_height - pad_top

    # padding
    img = cv2.copyMakeBorder(img,
                             pad_top,
                             pad_bottom,
                             pad_left,
                             pad_right,
                             cv2.BORDER_CONSTANT,
                             value=(0, 0, 0))

    try:
        if not(img.shape[0] == target_height and img.shape[1] == target_width):  # 둘 중 하나는 같아야 함
            raise Exception('Letter box resizing method has problem.')
    except Exception as e:
        print('Exception: ', e)
        exit(1)

    return img

class WindowClass(QMainWindow, form_class) :
    def __init__(self) :
        super().__init__()
        self.setupUi(self)
        
        self.video_capture = None
        
        #Icon Load
        self.pushButton_file_open.setIcon(QIcon(os.path.join(folder_abspath, 'images', 'icon_file_open.png')))
        
        self.pushButton_play.setIcon(QIcon(os.path.join(folder_abspath, 'images', 'icon_play.png')))
        self.pushButton_play.setIconSize(QSize(32, 32))

        self.pushButton_file_open.clicked.connect(self.load_video)
        self.pushButton_play.clicked.connect(self.play)
        self.pushButton_scene_start.pressed.connect(self.set_edes_frames)
        self.pushButton_play.setEnabled(False)
        self.pushButton_scene_start.setEnabled(False)

        self.horizontalSlider.setEnabled(False)
        self.horizontalSlider.valueChanged.connect(self.move_frame)

        self.scene_progressbar_timer = QTimer()
        
        self.video_play_timer = QTimer()
        self.video_play_timer.timeout.connect(self.read_next_frame)
        
        self.scene_progressbar_timer.setInterval(int(1000/60.))
        self.scene_progressbar_timer.timeout.connect(self.draw_scene_progress_bar)
        self.scene_progressbar_timer.start()

        QShortcut(Qt.Key_Space, self, self.play)

        self.listWidget.itemClicked.connect(self.move_scene)

        self.transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    def video_capture_release(self):
        if self.video_capture == None:
            return None
        self.video_capture.release()
    
    def load_video(self):
        
        self.video_file = QFileDialog.getOpenFileName(self, "Open a file", folder_abspath , "video file (*.mp4 *.avi *.mkv *.MP4 *.AVI *.MKV)")[0]
        print(self.video_file)
        if len(self.video_file) == 0:
            return None
        
        self.scene_start_frame_index = 0
        self.scene_end_frame_index = 0
        
        self.video_name = Path(self.video_file).resolve().stem
        self.patient_name = self.video_file.split('/')[-2]
        print(self.patient_name)
        self.frame_index = 0

        self.video_capture_release()
        self.video_capture = cv2.VideoCapture(self.video_file, apiPreference=cv2.CAP_FFMPEG)
        
        if self.video_capture == None or not self.video_capture.isOpened():
            return self.video_capture_release()
   
        self.video_fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        self.video_num_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.video_play_timer.setInterval(int(1000./self.video_fps))
                
        self.horizontalSlider.blockSignals(True)
        self.horizontalSlider.setValue(0)
        self.horizontalSlider.blockSignals(False)
        
        self.horizontalSlider.setEnabled(True)
        self.horizontalSlider.setMinimum(0)
        self.horizontalSlider.setMaximum(self.video_num_frames)
        
        self.pushButton_play.setEnabled(True)
        self.pushButton_scene_start.setEnabled(True)
        self.listWidget.clear()
        self.read_next_frame()

    def play(self):
        if not self.pushButton_play.isEnabled():
            return None
        
        if self.video_play_timer.isActive():
            self.pushButton_play.setIcon(QIcon(os.path.join(folder_abspath, "images", 'icon_play.png')))
            self.video_play_timer.stop()
        else:
            self.pushButton_play.setIcon(QIcon(os.path.join(folder_abspath, "images", 'icon_pause.png')))
            self.video_play_timer.start()
    
    def init_scene_setting(self):
        self.pushButton_scene_start.setEnabled(True)
        self.pushButton_scene_end.setEnabled(False)
        self.pushButton_scene_init.setEnabled(False)

    def set_scene_start_frame(self, index_frame):
        self.scene_start_frame_index = index_frame

    def set_scene_end_frame(self, index_frame):
        self.scene_end_frame_index = index_frame
        self.listWidget.addItem(str(self.scene_start_frame_index) + "_" + str(self.scene_end_frame_index))


    def move_scene(self):
        if self.video_play_timer.isActive():
            self.video_play_timer.stop()
        
        clicked_item = self.listWidget.currentItem().text()
        clicked_item = clicked_item.split('_')
            
        start_frame_index = int(clicked_item[0])
        
        self.frame_index = start_frame_index
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)
        
        self.read_next_frame()
        
        print(str(self.listWidget.currentRow()) + " : " + self.listWidget.currentItem().text())

    def remove_scene(self):
        self.removeItemRow = self.listWidget.currentRow()
        self.listWidget.takeItem(self.removeItemRow)

    def cutImg(self, img_raw):
        img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2HSV)
        img_h = img[:, :, 0]
        img_s = img[:, :, 1]
        img_v = img[:, :, 2]
        img_v[img_v > 20] = 255
        img_v[img_v < 50] = 0
        img_label = label(img_v, connectivity=2)
        props = regionprops(img_label)

        area_list = []
        label_list = []
        for i in range(len(props)):
            area_list.append(props[i].area)
            label_list.append(props[i].label)

        area_index = area_list.index(max(area_list))
        box_loc = props[area_index].bbox
        img_cut = img_raw[box_loc[0]:box_loc[2], box_loc[1]:box_loc[3], :]
        return img_cut

    def load_weights(self, networks, filename = ''):
        if os.path.isfile(filename):
            net_state_file = torch.load(filename)
            networks.load_state_dict(net_state_file['network'])
            print('save network epoch of {}'.format(net_state_file['epoch']))
    def set_edes_frames(self):
        transformTest = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        tensors = {}
        tensors['dataX'] = torch.FloatTensor()
        networks = importlib.import_module('architectures.DenseNet').create_model().cuda()
        networks.eval()
        self.load_weights(networks, filename = './model_DenseNet')
        net_coll = importlib.import_module('architectures.mobileVit').create_model().cuda()
        net_coll.eval()
        self.load_weights(net_coll, filename = './model_mobileVit')

        cam = cv2.VideoCapture(self.video_file)
        pred_cls = []
        while True:
            res, image = cam.read()
            if res == True:
                # image = cv2.imread(os.path.join(path_img, img_name))
                img_cut = self.cutImg(image)
                # print(img_cut.shape)
                img_cut = cv2.cvtColor(img_cut, cv2.COLOR_BGR2RGB)
                img_cut = Image.fromarray(img_cut)
                # img_cut = Image.open(os.path.join(path_img, img_name))
                input_tensor = transformTest(img_cut)
                input_tensor = torch.unsqueeze(input_tensor, dim=0).cuda()
                tensors['dataX'].resize_(input_tensor[0].size()).copy_(input_tensor[0])
                with torch.no_grad():
                    dataX_var = torch.autograd.Variable(input_tensor)
                pred_var = networks(dataX_var)
                pred_pob = nn.Softmax(dim=1)(pred_var)
                pred_var_coll = net_coll(dataX_var)
                pred_pob_coll = nn.Softmax(dim=1)(pred_var_coll)
                predConf = (torch.abs(pred_pob[0][1] - 0.5) * 2).item()
                predCollConf = (torch.abs(pred_pob_coll[0][1] - 0.5) * 2).item()
                if predConf < 0.35:  #
                    if predCollConf > predConf:
                        pred_var[0] = (pred_pob_coll[0] + pred_pob[0])/2
                _, pred = pred_var.topk(1, 1, True, True)
                # print(pred)
                pred = pred.t()[0][0].item()
                pred_cls.append(pred)
                # print(pred)
            else:
                break
        print(pred_cls)
        pred_key = []
        for i in range (len(pred_cls)):
            if pred_cls[i] == 1:
                pred_key.append(i+1)
        print(pred_key)
        pred_cut_index = []
        for i in range (len(pred_key)-1):
            if pred_key[i+1] -pred_key[i] != 1:
                pred_cut_index.append(i)
        pred_cut_index.append(len(pred_key)-1)
        pred_cut_index.insert(0,0)
        for i in range(len(pred_cut_index)-1):
            if i == 0:
                self.set_scene_start_frame(pred_key[0])
                self.set_scene_end_frame(pred_key[pred_cut_index[i+1]])
            else:
                self.set_scene_start_frame(pred_key[pred_cut_index[i]+1])
                self.set_scene_end_frame(pred_key[pred_cut_index[i+1]])

    def move_frame(self):
        if self.video_play_timer.isActive():
            self.video_play_timer.stop()
            
        self.frame_index = self.horizontalSlider.value()-1
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)
        
        self.read_next_frame()
        
        if not self.pushButton_scene_start.isEnabled():#
            if self.frame_index < self.scene_start_frame_index:
                QMessageBox.question(self, 'Message', 'Scene setting is intialized', QMessageBox.Yes)
                self.init_scene_setting()
            
    def read_next_frame(self):
        if not self.pushButton_play.isEnabled():
            return None
        
        read_frame, frame = self.video_capture.read()
        
        if read_frame:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = letter_box_resize(frame, (self.label_frame.width(), self.label_frame.height()))
            height, width, channels = frame.shape
            bytesPerLine = channels * width
            qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap01 = QPixmap.fromImage(qImg)
            
            self.label_frame.setPixmap(pixmap01)
            self.frame_index += 1

            self.horizontalSlider.blockSignals(True)
            self.horizontalSlider.setValue(self.frame_index)
            self.horizontalSlider.blockSignals(False)
            
            self.label_frame_index.setText(str(self.frame_index)+ "/" + str(self.video_num_frames))

    def draw_scene_progress_bar(self):
        if not self.pushButton_play.isEnabled():
            return None
        
        scene_progress_bar = np.zeros((self.label_scene_progress_bar.height(), self.label_scene_progress_bar.width(), 3), np.uint8)
        
        if not self.pushButton_scene_start.isEnabled():
            start_frame_index = int(self.label_scene_progress_bar.width() * (float(self.scene_start_frame_index) / self.video_num_frames))
            end_frame_index = int(self.label_scene_progress_bar.width() * (float(self.frame_index) / self.video_num_frames))
            scene_progress_bar[:, start_frame_index:end_frame_index] = [255, 255, 0]
            
        for item_index in range(self.listWidget.count()):
            item = self.listWidget.item(item_index).text()
            item = item.split('_')
            
            start_frame_index = int(self.label_scene_progress_bar.width() * (float(item[0]) / self.video_num_frames))
            end_frame_index = int(self.label_scene_progress_bar.width() * (float(item[1]) / self.video_num_frames))
            
            scene_progress_bar[:, start_frame_index:end_frame_index] = [0, 255, 0]

        height, width, channels = scene_progress_bar.shape
        bytesPerLine = channels * width
        qImg = QImage(scene_progress_bar.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap01 = QPixmap.fromImage(qImg)
        self.label_scene_progress_bar.setPixmap(pixmap01)

if __name__ == "__main__" :
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()

