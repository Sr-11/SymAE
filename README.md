# Question:  
0. questions. 
## 6/2  
1. If we use sigma=0.1 to train the NN, then use sigma=0.0 to test it, the NN will "learn the noise"?    
2. We are over fitting. g0-g9 is not enough. If input generate(random), output still like g's.    
## 6/5  
3. If we only use cos(x)...cos(10x) to train SymAE, can't deal with cos(100x)?
4. Coherent code is translation-invarant
5. How do we know whether a NN is convex wrt all parameters? what if it never reached the global minimun
6. I believe there is no way to catch the high frequency information？ Like the theta is completely random and our training set contains smooth functions  
## 6/6  
7. Another issue is my code is not good at generalization. The quality is very bad for inputs that are not in my training set.
8. generate_trigonometric isn't enough. because the NN is not linear. it's even not enough by a scale.
9. Looks the training process is much more sensitive to high frequency information? This may be due to some internal structure of symae_core. No is it a optical illusion？
10. Thm3.1 in MRA can be stronger. Cd is not depending on d.
11. There is a typo in MRA paper: Thm A1, proof, line 3.
12. KL distance and $\chi^2$ distance.
## 6/7
13. B's manuscript. Definition of A specifies rank(X)=1, while Lemma5.1 rank(X)=1 or 2.
14. Why $|S_\epsilon|\leq\frac{Vol(S+\frac12D)}{Vol(\frac12D)}$? In "Tight Oracle Inequalities," beginning of III.PROOFS
15. Lemma3.1(Covering Number for Low-Rank Matrices) in "Tight Oracle Inequalities" can be stronger, r->r-1
16. How to def $Vol()$ ? In proof of Lemma3.1
17. 