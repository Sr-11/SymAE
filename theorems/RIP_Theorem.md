## Proposition 4 (Covering Number)
(https://www.math.uci.edu/~rvershyn/teaching/2006-07/280/lec6.pdf)  
$$N(K,D)\leq\frac{Vol(K+\frac12D)}{Vol(\frac12D)}$$
where $N(K,D):=$ minimal number of translates of D to cover K.
#### Proof.
Start with an arbitrary point $x_1 \in K$.  
Then choose $x_2 \in K$: $||x_2 - x_1|| > \varepsilon$.  
Then choose $x_3 \in K$: $||x_3 - x_{1,2}|| > \varepsilon$.  
......  
Then choose $x_N \in K$: $||x_N - x_{1\sim N-1}|| > \varepsilon$.  
$\underline{Claim}$:  
$\{x_1,...x_N\}$ is an $\varepsilon-net$.  
Suppose not. Then $\exists z \in K$ s.t. $||z - x_{1\sim N}|| > \varepsilon$.  
Recall that each pair $(x_i,x_j)$ is at least  $\varepsilon$-apart.  
Then, if we shrink the radius of these balls to $\frac\varepsilon2$, they will be disjoint.  
$$Vol(K+\frac12D)\geq N\cdot Vol(\frac12D)$$

## Theorem 2.3 
(Tight Oracle Inequalities for Low-Rank Matrix
Recovery From a Minimal Number of
Noisy Random Measurements)  
Fix $0\leq\delta<1$.  
$A:\mathbb{R}^{n_1\times n_2}\to\mathbb{R}^n$ is a random measurement ensemble obeying the following condition.  
$\forall X \in \mathbb{R}^{n_1\times n_2}, \forall t\in(0,1)$  
$\exists C,c>0, s.t.$  
$$P(|||A(x)||_2^2-||X||_F^2|>t||X||_F^2)\leq Ce^{-cm}$$
Then, $\exists D,d$, s.t.  
$\forall m\geq Dnr$  
$A$ satisfies the RIP with isometry constant with probability exceeding $1-Ce^{-dm}$ .