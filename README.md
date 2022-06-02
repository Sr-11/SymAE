# Question:  
1. Why "Num GPUs Available:  0" ? Wave3 is supposed to have 1 gpu. Run on wave3, see src/main.ipynb
2. I can't use wave1.mit.edu ??
(base) sunrui@dhcp-10-31-48-146 ~ % ssh -XC eruisun@wave1.mit.edu
kex_exchange_identification: read: Connection reset by peer
3. What's symae_core doing? 
What I see is that it is just simply defining various NeuralNets. 
I can't see where is the structure of SymAE?
So we need to construct SymAE by hand?
# SymAE+MRA  
Generating MRA data: src/generate.py  
![Alt text](symae.png?raw=true "Title")
## How to write code on the server:
conda activate MRA 
jupyter lab --allow-root --ip=0.0.0.0 --no-browser  (must under the home directory)  
(wave3 is strange. Must replace "tswave3" with "wave3.mit.edu")  
## How to use git on the server:
**Note! All the operations must under /eruisun/github/**  
use pwd to check  
**How to change repository?**  
git remote set-url origin git@github.com:Sr-11/SymAE.git  
git remote -v  
**How to push from server wave1?**  
1.jupyter lab --allow-root --ip=0.0.0.0 --no-browser  
2.open jupyter lab in browser locally  
3.manage code  
4.git commit -a (use vim)  
5.git push  
4.git add .  
5.git status -s  
6.git commit -m "comments"  
7.git status  
8.git push  
**How to pull?**    
git pull = git fetch + git merge  
## How to transport files using SSH?
scp /Users/sunrui/Desktop/symae.png eruisun@wave1.mit.edu:/math/home/eruisun  
