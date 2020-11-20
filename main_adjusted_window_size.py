#%%
import numpy as np
import cv2
import matplotlib.pyplot as plt
import win32gui,win32ui,win32con
import pyautogui as pg
from PIL import ImageGrab
import torch
import time
import random
from click_buttons import click_button,click_button_relative,states_str,states_num
# %%
def get_window(lpClassName='UnityWndClass', lpWindowName='炉石传说'):
    handle_of_hearthstone=win32gui.FindWindow(lpClassName,lpWindowName)
    return win32gui.GetClientRect(handle_of_hearthstone)
def countdown(n,printnum=True):
    for i in np.arange(n,0,-1):
        if printnum:    
            print(i)
        time.sleep(1)
print('You have 10 seconds to move mouse to the corner')
countdown(10)
corner=np.array([0,42])
new_total=np.array(pg.position())-corner
window=np.array(get_window())
window[:2]+=corner
window[2:]+=corner
window=tuple(window)
def currentmouse():
    return pg.position()
def get_pic():
    return np.array(ImageGrab.grab(window))
def closewindow():
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# %%
# cv2.imshow('22',get_pic())
# closewindow()

#%% 读取alexnet
from net import AlexNet,preprocess_image,MyDataset
alexnet=AlexNet()
state_dict=torch.load('./params_alexnet/alexnet_adam_batch_size=4_epochs=20_lr=1e-05_channels=96_256_300_256_19712_2000_2000_7')
alexnet.load_state_dict(state_dict)

# %%
def predict_state(pic):
    pic=preprocess_image(pic).view(1,3,497,662)
    with torch.no_grad():
        pre=alexnet(pic)
    return pre.argmax(axis=1).item()
# %%
print('You have 5 seconds to get ready')
countdown(5)
new_total=np.array(get_window())[2:]
first_turn=0
expression=['感谢',
    '称赞',
    '问候',
    '惊叹',
    '失误',
    '威胁']
states_str=['主界面','选牌界面','战斗界面','收藏界面','搜索界面','手牌更换','战斗结果']
while(True):
    
    pic=get_pic()
    # cv2.imshow('copy of hearthstone',pic)
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    #     break
    pic=cv2.resize(pic,(662,497))
    state_num=predict_state(pic)
    state_str=states_str[state_num]
    print('state',state_str)
    if state_num==0:
        click_button_relative('主界面','对战模式',new_total=new_total,sleep=False)
    if state_num==1:
        click_button_relative('选牌界面','开始',new_total=new_total,sleep=False)
        first_turn=0
    if state_num==2:
        click_button_relative('战斗界面','技能',new_total=new_total,sleep=True)
        click_button_relative('战斗界面','我方脸',new_total=new_total,sleep=False,right_click=True)
        time.sleep(0.5)
        click_button_relative('战斗界面','感谢',new_total=new_total,sleep=False)
        #click_button_relative('战斗界面','设置',new_total=new_total,sleep=True)
        #click_button_relative('战斗界面','认输',new_total=new_total,sleep=True)
        time.sleep(np.random.randint(low=0,high=6))
        click_button_relative('战斗界面','我方脸',new_total=new_total,sleep=False,right_click=True)
        time.sleep(0.5)
        click_button_relative('战斗界面',expression[np.random.randint(low=0,high=6)],new_total=new_total,sleep=False)
        
        click_button_relative('战斗界面','回合结束',new_total=new_total,sleep=True)
        first_turn+=1
    if state_num==3:
        pass
    if state_num==4:
        pass
    if state_num==5:
        click_button_relative('手牌更换','确认',new_total=new_total,sleep=True)
        first_turn=0
    if state_num==6:
        click_button_relative('战斗结束','确认',new_total=new_total,sleep=True)
    click_button_relative('其他','乱按',new_total=new_total,sleep=True)
    """if first_turn==2:
        click_button_relative('战斗界面','对方脸',new_total=new_total,sleep=False,right_click=True)
        click_button_relative('战斗界面','屏蔽对方',new_total=new_total,sleep=True)"""
    
    click_button_relative('战斗界面','回合结束',new_total=new_total,sleep=True)
    countdown(np.random.randint(low=3,high=10),printnum=False)

# %%
