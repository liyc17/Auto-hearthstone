#%%
import numpy as np
import time
import  pyautogui as pg
# %%
states_str=['主界面','选牌界面','战斗界面','收藏界面','搜索界面','手牌更换','战斗结果']
states_num=[0,1,2,3,4,5,6]
corner=np.array([0,42])
click_positions_absolute={'total':np.array([662,497])}
click_positions_absolute['主界面']={
    '对战模式':np.array([332,148])
}
click_positions_absolute['选牌界面']={
    '返回':np.array([623,456]),
    '开始':np.array([535,406]),
    '狂野':np.array([536,22]),
    '休闲':np.array([491,73]),
    '排名':np.array([579,73])
}
click_positions_absolute['战斗界面']={
    '技能':np.array([413,376]),
    '回合结束':np.array([602,226]),
    '认输':np.array([332,171]),
    '设置':np.array([644,482]),
    '手牌':np.array([150,416,485,493]),
    '我方随从':np.array([74,227,566,316]),
    '对方随从':np.array([74,143,560,230]),
    '我方脸':np.array([334,411]),
    '感谢':np.array([262,352]),
    '称赞':np.array([245,387]),
    '问候':np.array([248,429]),
    '惊叹':np.array([405,348]),
    '失误':np.array([428,386]),
    '威胁':np.array([427,433]),
    '对方脸':np.array([334,125]),
    '屏蔽对方':np.array([259,72])
}
click_positions_absolute['收藏界面']={

}
click_positions_absolute['搜索界面']={
    '取消':np.array([342,417])
}
click_positions_absolute['手牌更换']={
    '确认':np.array([333,389])
}
click_positions_absolute['战斗结束']={
    '确认':np.array([306,61])
}
click_positions_absolute['其他']={
    '乱按':np.array([91,57])
}
def click_button(state,substate,position_harsh=click_positions_absolute,sleep=True):
    if sleep:
        time.sleep(1+np.random.rand())
    position=position_harsh[state][substate]+corner
    pg.moveTo(position[0],position[1])
    pg.click()

def click_button_relative(state,substate,new_total,position_harsh=click_positions_absolute,sleep=True,right_click=False):
    if sleep:
        time.sleep(1+np.random.rand())
    position=(position_harsh[state][substate])/position_harsh['total']*np.array(new_total)+corner
    pg.moveTo(position[0],position[1])
    if right_click:
        pg.rightClick()
    else:
        pg.click()
# %%
