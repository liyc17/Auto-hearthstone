#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
from torch.utils.tensorboard import SummaryWriter
# %%    
scale=1
def preprocess_image(pic,intorch=True):
    if intorch:
        return torch.from_numpy(pic).permute(2,0,1).type(torch.float)
    else:
        return pic
class MyDataset(torch.utils.data.Dataset):
    def __init__(self,intorch=True):
        self.path='./dataset/'
        self.all_pic_name=os.listdir(self.path)
        self.intorch=intorch
    def __len__(self):
        return len(self.all_pic_name)
    def __getitem__(self,index):
        label=int(self.all_pic_name[index][0])
        pic=cv2.imread(self.path+self.all_pic_name[index])/scale
        #pic=torch.from_numpy(pic).permute(2,0,1).type(torch.float)
        pic=preprocess_image(pic,intorch=self.intorch)
        return (pic,label)
class AlexNet(nn.Module):  
    def __init__(self):
        super(AlexNet,self).__init__()
        self.channel1=96
        self.channel2=256
        self.channel3=300
        self.channel4=256
        self.channel5=self.channel4*7*11
        self.channel6=2000
        self.channel7=2000
        self.channel8=7
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=self.channel1,kernel_size=11,stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=self.channel1,out_channels=self.channel2,kernel_size=5,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=self.channel2,out_channels=self.channel3,kernel_size=3),
            nn.ReLU(),
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(in_channels=self.channel3,out_channels=self.channel4,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=3),
        )
        self.fc1=nn.Sequential(
            nn.Linear(in_features=self.channel5,out_features=self.channel6),
            nn.ReLU()
        )
        self.fc2=nn.Sequential(
            nn.Linear(in_features=self.channel6,out_features=self.channel7),
            nn.ReLU()
        )
        self.fc3=nn.Sequential(
            nn.Linear(in_features=self.channel7,out_features=self.channel8)
        )
        

    def forward(self,x):
        conv1_out=self.conv1(x)
        #print(conv1_out.size())
        conv2_out=self.conv2(conv1_out)
        #print(conv2_out.size())
        conv3_out=self.conv3(conv2_out)
        #print(conv3_out.size())
        conv4_out=self.conv4(conv3_out)
        #print(conv4_out.size())
        conv4_out=conv4_out.view((conv4_out.size()[0],-1))
        #print(conv4_out.size())
        fc1_out=self.fc1(conv4_out)
        fc2_out=self.fc2(fc1_out)
        fc3_out=self.fc3(fc2_out)
        #print(fc3_out.size())
        return fc3_out
#%%
if __name__=='__main__':
    alexnet=AlexNet()
    if torch.cuda.is_available():
        alexnet.cuda()
    criterion=torch.nn.CrossEntropyLoss()
    lr=0.0001
    optimizer=torch.optim.Adam(params=alexnet.parameters(),lr=lr,betas=(0.9,0.99),eps=1e-7)
    #optimizer=torch.optim.SGD(params=alexnet.parameters(),lr=0.01,momentum=0.9)

    # a=cv2.imread('./dataset/0_1.png')
    # print(a.shape)
    # a=torch.from_numpy(a).permute(2,0,1).type(torch.float)
    # plt.imshow(a.permute(1,2,0).type(torch.int).numpy())
    # print(a.size())
    # with torch.no_grad():
    #     out=alexnet(a.view(1,a.size(0),a.size(1),a.size(2)))
    #     print(out.size())

    # %%
    

    #%% 读取数据 定义参数
    batch_size=4
    epochs=20
    epoch_lapse=5
    mydataset=MyDataset()
    #%%
    modulename='alexnet_adam_batch_size={}_epochs={}_lr={}_channels={}_{}_{}_{}_{}_{}_{}_{}_scale={}'.format(batch_size,epochs,lr,alexnet.channel1,alexnet.channel2,alexnet.channel3,alexnet.channel4,alexnet.channel5,alexnet.channel6,alexnet.channel7,alexnet.channel8,scale)
    writer=SummaryWriter('./runs/'+modulename) #comment=modulename)

    a=cv2.imread('./dataset/0_1.png')
    a=torch.from_numpy(a).permute(2,0,1).type(torch.float)
    with torch.no_grad():
        writer.add_graph(alexnet,a.view(1,a.size(0),a.size(1),a.size(2)).cuda())
    for epoch in range(epochs):
        running_loss=0
        total_loss=0
        accurate=0
        pic_num=0
        correct_num=0
        train_loader=torch.utils.data.DataLoader(mydataset,batch_size=batch_size,shuffle=False)
        train_iter=iter(train_loader)
        for i in range(len(train_iter)):
            pics,labels=train_iter.next()
            #labels=torch.tensor(labels)
            if torch.cuda.is_available():
                pics=pics.cuda()
                labels=labels.cuda()
            output=alexnet(pics)
            loss=criterion(output,labels)
            loss.backward()
            optimizer.step()

            running_loss+=loss.item()
            total_loss+=loss.item()
            predict=output.argmax(axis=1)
            pic_num+=predict.size(0)
            correct_num+=sum(predict==labels).item()
            accurate=correct_num/pic_num
            if (i+1)%epoch_lapse==0:
                print('picture',pic_num,end='\t')
                print('running loss {:.3f}'.format(running_loss),end='\t')
                print('current loss {:.3f}'.format(loss.item()),end='\t')
                print('accurate {:.3f}'.format(accurate))
                running_loss=0
            writer.add_scalar(tag='accurate',scalar_value=accurate,global_step=epoch*mydataset.__len__()+pic_num)
            writer.add_scalar(tag='running loss',scalar_value=running_loss,global_step=epoch*mydataset.__len__()+pic_num)
            writer.add_scalar(tag='total loss',scalar_value=total_loss,global_step=epoch*mydataset.__len__()+pic_num)
        print('finish epoch',epoch+1)
    writer.close()

    torch.save(alexnet.state_dict(),'./params_alexnet/{}'.format(modulename))
# %%
