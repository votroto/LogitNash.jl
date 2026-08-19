The input is a multiplayer strategic game defined by utility tensors $(u_p)_{p \in N}$. Player $p$ has actions $1,\dots,D_p$. Let $\pi_p^i$ denote the probability that player $p$ plays action $i$, and let $\mu_p^i$ be the corresponding reduced log-ratio coordinate, with action $D_p$ as reference.

The unilateral deviation utility of player $p$ to action $i$ is

$$U_p^i(\pi_{-p})=\sum_{a_{-p}} u_p(i,a_{-p}) \prod_{q\neq p}\pi_q^{a_q}$$


The partial derivative with respect to the probability that player $q$ plays action $j$ is

$$\frac{\partial U_p^i}{\partial \pi_q^j}=\mathbf 1_{{p\neq q}} \sum_{a_{-(p,q)}} u_p(i,j,a_{-(p,q)}) \prod_{r\neq p,q}\pi_r^{a_r}$$

## Continuation

A logit equilibrium parametrized by precision $t$ satisfies

$$ \pi_p^i = \frac{\exp(tU_p^i(\pi_{-p}))}{\sum_j \exp(tU_p^j(\pi_{-p}))} $$

The algorithm starts with a uniform profile at $t=0$ and tracks the solution path towards infinity.

## Residual
For $i < D_p$

$$F_p^i(\mu, t) = \mu_p^i - t(U_p^i(\pi_{-p}) - U_p^{D_p}(\pi_{-p}))$$

and its partial derivatives wrt. $\mu$ are

$$ \frac{\partial F_p^i}{\partial\mu_q^j} = -t\pi_q^j \left[ \left( \frac{\partial U_p^i}{\partial\pi_q^j} - \frac{\partial U_p^{D_p}}{\partial\pi_q^j} \right) - \sum_{k=1}^{D_q} \left( \frac{\partial U_p^i}{\partial\pi_q^k} - \frac{\partial U_p^{D_p}}{\partial\pi_q^k} \right)\pi_q^k \right] $$

when $p\neq q$, or the identity matrix otherwise; while wrt. $t$ the formula is

$$ \frac{\partial F_p^i}{\partial t} = U_p^{D_p}-U_p^i. $$

## Predictor

Using previous tangents $d\mu_\text{last}$ and $dt_\text{last}$ as a guide, the current tangent $(d\mu, dt)$ is obtained from

$$\begin{bmatrix} F_\mu & F_t\\ d\mu_\text{last}' & dt_\text{last} \end{bmatrix} \begin{bmatrix} d\mu\\ dt \end{bmatrix}= \begin{bmatrix} 0\\ 1 \end{bmatrix}$$

giving the Euler prediction for a step size $\Delta s$ of

$$\begin{aligned} \hat{\mu} &= \mu + d\mu\Delta s \\ \hat{t} &= t + dt\Delta s \end{aligned}$$

## Pseudo-Arclength Constraint

Given the prediction $(\hat{\mu}, \hat{t})$, the arclength constraint is

$$ g(\mu,t) = (\mu-\hat{\mu})'d\mu + (t - \hat{t})dt = 0 $$

with the partial derivatives


$$\begin{aligned} g_\mu &= d\mu'\\ g_t &= dt \end{aligned}$$

## Corrector

The newton corrections $\mu \gets \mu + \Delta\mu$, $t \gets t + \Delta t$ are given by

$$\begin{bmatrix} F_\mu & F_t\\ g_\mu & g_t \end{bmatrix} \begin{bmatrix} \Delta \mu\\ \Delta t \end{bmatrix}=- \begin{bmatrix} F\\ g \end{bmatrix}$$

