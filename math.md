The input is a multiplayer strategic game defined by utility tensors $(u_p)_{p \in N}$. Player $p$ has actions $1,\dots,D_p$. Let $\pi_p^j$ denote the probability that player $p$ plays action $i$, and let $\mu_p^i$ be the corresponding reduced log-ratio coordinate, with action $D_p$ as reference.

The unilateral deviation utility of player $p$ to action $i$ is

$$
U_p^i(\pi_{-p})
=
\sum_{a_{-p}} u_p(i,a_{-p}) \prod_{q\neq p}\pi_q^{a_q}
$$

The derivative with respect to the probability that player $q$ plays action $j$ is

$$
\frac{\partial U_p^i}{\partial \pi_q^j}
=
\mathbf 1_{{p\neq q}} \sum_{a_{-(p,q)}} u_p(i,j,a_{-(p,q)}) \prod_{r\neq p,q}\pi_r^{a_r}
$$

## Residual
For $i < D_p$

$$
F_p^i(\mu, t) = \mu_p^i - t(U_p^i(\pi_{-i}) - U_p^{D_p}(\pi_{-i}))
$$

## Predictor

The tangent $d\pi, dt$ is obtained from


$$
\begin{bmatrix}
F_\mu & F_t\\
d\mu_\text{last}' & dt_\text{last}
\end{bmatrix}
\begin{bmatrix}
d\mu\\
dt
\end{bmatrix}=
\begin{bmatrix}
0\\
1
\end{bmatrix}
$$

giving the prediction $\hat{\mu} = \mu + ds\,d\pi$, $\hat{t} = t + ds\,dt$.

## Corrector

The corrector solves

$$
\begin{cases}
F(\mu,t) = 0\\
(\mu-\hat{\mu})'d\mu + (t - \hat{t})dt = 0
\end{cases}
$$

with Newton corrections $\mu \gets \mu + d\mu$, $t \gets t + dt$,  given by
$$
\begin{bmatrix}
F_\mu & F_t\\
g_\mu & g_t
\end{bmatrix}
\begin{bmatrix}
d\mu\\
dt
\end{bmatrix}=
\begin{bmatrix}
F\\
g
\end{bmatrix}
$$

