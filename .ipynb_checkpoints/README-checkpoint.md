# Question: 
## 6/2
1. If we use sigma=0.1 to train the NN, then use sigma=0.0 to test it, the NN will "learn the noise"?  
2. We are over fitting. g0-g9 is not enough. If input generate(random), output still like g's.  
## 6/5
3. If we only use cos(x)...cos(10x) to train SymAE, can't deal with cos(100x)?
4. Coherent code is translation-invarant
5. How do we know whether a NN is convex wrt all parameters? what if it never reached the global minimun
6. I believe there is no way to catch the high frequency information？ Like the theta is completely random and our training set contains smooth functions
7. Another issue is my code is not good at generalization. The quality is very bad for inputs that are not in my training set.
8. generate_trigonometric isn't enough. because the NN is not linear. it's even not enough by a scale.
9. Looks the training process is much more sensitive to high frequency information? This may be due to some internal structure of symae_core. No is it a optical illusion？
10. 