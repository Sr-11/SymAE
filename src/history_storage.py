import os
import re
import datetime
import MRA_generate as generate
import matplotlib.pyplot as plt
import shutil
class history_storage():
    def __init__(self,model,sigma,epochs,p,q,d,nt):
        self.model=model
        self.sigma=sigma
        self.epochs=epochs
        self.p=p
        self.q=q
        self.d=d
        self.nt=nt
    def find_file(self,start,target):
        list=[]
        for relpath, dirs, files in os.walk(start):
            for name in files:
                searchObj=re.search(target,name)
                if searchObj:
                    list.append(searchObj)
        return list
    def find_dirs(self,start,target):
        list=[]
        for relpath, dirs, files in os.walk(start):
            for name in dirs:
                searchObj=re.search(target,name)
                if searchObj:
                    list.append(searchObj)
        return list
    def list_blank(self,list):
        list.sort()
        for i in range(len(list)):
            if list[i]!=i+1:
                return i+1
        return len(list)+1
    def list_up(list):
        list.sort()
        return list[-1]+1
    def save_figure(self,fig):
        list=findfile('./figures',r'_(\d)\.png')
        list.sort()
        k=len(list)+1
        for i in range(len(list)):
            if list[i]!=i+1:
                k=i+1
        print('There are %d figures in src/plots'%(k-1))
        fig.savefig('./plots/result_%d.png'%k)
    def save_weights(self,ID):
        for relpath, dirs, files in os.walk('./checkpoints'):
            for name in dirs:
                searchObj=re.search(r'ID=.*',name)
                if searchObj:
                    full_path = os.path.join('./checkpoints', name)
                    shutil.retree(full_path)
        date_time=datetime.datetime.now().strftime("%m-%d-%Y--%H-%M-%S")
        file_name='ID={},sigma={},epochs={},p={},q={},date={}'.format(ID,self.sigma,self.epochs,self.p,self.q,date_time)
        self.model.save_weights('./checkpoints/'+file_name)
    def save_visualize(self,ID):
        test_X=generate.generate_smooth(self.d,self.nt,1,10,0)
        test_Y=self.model.predict(test_X)
        plt.rc('font', size=20)
        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12,6))
        fig.suptitle('p=%d, q=%d'%(self.p,self.q))
        axs[0].set_title('input')
        axs[1].set_title('output')
        for j in range(5):
            axs[0].plot(range(self.d),test_X[0,j,:],label='%d'%j)
        for j in range(5):
            axs[1].plot(range(self.d),test_Y[0,j,:],label='%d'%j)
        for ax in axs.flat:
            ax.grid(True)
            ax.set(xlabel='x',ylabel='value')
            ax.legend()
        fig.savefig('./figures/ID_%d.png'%ID)
    def save_everything(self,ID):
        self.save_weights(ID)
        self.save_visualize(ID)  
        print("===== Save Complete =====")
    def load_weights(self,ID):
        full_path=''
        for relpath, dirs, files in os.walk('./checkpoints'):
            for name in dirs:
                searchObj=re.search(r'ID=.*',name)
                if searchObj:
                    full_path = os.path.join('./checkpoints', name)
        self.model.load_weights(full_path)
