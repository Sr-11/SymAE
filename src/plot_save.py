import os,re
def plot_save(fig,ID):
    '''
    Save the fig as a file
    
    Parameters
    ----------
    fig : A plt figure object
    ID : int
        If ID>=1, it's used to distinguish different plots, each .png file has a unique ID>=1.
        If ID>=1 and this ID has been used, fig will saved as './plots/plot_0.png'
        If ID==-1, this function will find the smallest available (hasn't been used) ID.
        If ID==0, './plots/plot_0.png' is a temporary storage.
        
    Yields
    ----------
    A figure saved in ./plots, named as plot_ID.png
    '''
    # Check if plot_ID exist
    if ID==-1:
        list=[]
        for relpath, dirs, files in os.walk('./plots'):
            for name in files:
                searchObj=re.search('plot_(\d*)\.png',name)
                if searchObj and int(searchObj.group(1))!=0:
                    list.append(int(searchObj.group(1)))
        list.sort()
        blank=len(list)+1
        for i in range(len(list)):
            if list[i]!=i+1:
                blank=i+1
        fig.savefig('./plots/plot_%d.png'%blank)
        print('Saved as plot_%d.png'%blank)
    else:
        for relpath, dirs, files in os.walk('./plots'):
            for name in files:
                searchObj=re.search('plot_%d.png'%ID,name)
                if searchObj:
                    print('plot_%d.png already exist'%ID)
                    #print('Saved as plot_0.png')
                    #ID=0
                    break
        # Save plot
        fig.savefig('./plots/plot_%d.png'%ID)