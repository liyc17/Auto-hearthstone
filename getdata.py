#%%
import numpy as np
import cv2
import matplotlib.pyplot as plt
import win32gui,win32ui,win32con,win32api 
import pyautogui as pg
from PIL import ImageGrab
import time
import pandas as pd

# %%
def get_window(lpClassName='UnityWndClass', lpWindowName='炉石传说'):
    handle_of_hearthstone=win32gui.FindWindow(lpClassName,lpWindowName)
    return win32gui.GetClientRect(handle_of_hearthstone)
def countdown(n):
    for i in np.arange(n,0,-1):
        print(i)
        time.sleep(1)
countdown(5)
corner=pg.position()
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


#%%
states_str=['主界面','选牌界面','战斗界面','收藏界面','搜索界面','手牌更换','战斗结果']
states_num=[0,1,2,3,4,5,6]
states=pd.DataFrame(states_str,index=states_num)
print(states)
count=np.load('count.npy')
while(True):
    pic=get_pic()
    cv2.imshow('output',pic)
    key=chr(cv2.waitKey(0))
    cv2.destroyAllWindows()
    if key=='q':#quit
        break
    elif key=='d':#discard
        pass
        countdown(5)
    else:
        count+=1
        plt.imsave('./dataset/{}_{}.png'.format(key,count[0]),pic)
        countdown(5)
np.save('count.npy',count)

#%% 收集按钮位置
if False:
    countdown(5)
    print(pg.position())
    print(np.array(get_window()))
    print(np.array(pg.position())-corner)