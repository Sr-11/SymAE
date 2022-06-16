import os
import re
def findfile(start):
    list=[]
    for relpath, dirs, files in os.walk(start):
        for name in files:
            searchObj=re.search(r'_(\d)\.png',name)
            if searchObj:
                list.append(int(searchObj.group(1)))
                #full_path = os.path.join(start, relpath, name)
                #print(os.path.normpath(os.path.abspath(full_path)))
    return list
def findblank(list):
    list.sort()
    for i in range(len(list)):
        if list[i]!=i+1:
            return i+1
    return len(list)+1
def savefig(fig):
    list=findfile('./plots')
    print(list)
    i=findblank(list)
    print('There are %d figures in src/plots'%(i-1))
    fig.savefig('./plots/result_%d.png'%i)