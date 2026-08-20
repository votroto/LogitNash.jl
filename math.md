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

## Algorithm Design

1. The number of Newton corrections is kept low deliberately to prevent path jumping and keep the runtime low. This is an effective and deliberate choice inspired by HomotopyContinuation.jl and Bertini.
2. Path jumping is prevented using a determinant check. The sign should be invariant along a smooth path segment, however we still have to be able to cross bifurcations which are inevitably present in many famous traditional games. Forcing the determinant to maintain orientation will stall progress. An angular distance check is ineffective for detecting jumps. For example, when the path follows an $\Omega$ shaped section, we can go around the first tight bend, gain speed on the smooth arc and then jump across. The determinant will flip as we are now tracking backwards, but the angular distance may be arbitrarily small as the point on the backwards path may even come directly from the predictor.
3. The step size must remain unbounded as we need to reach a precision of $10^6$ quickly. Forcing $\Delta s \leq 1$ might solve one in a million particurarily evil logit paths, but at the cost of unnecessarily increasing the number of iterations $1000\times$ for the rest.
4. Using a stale or approximate Jacobians for predictions is unacceptable in general. Paths occasionally contain extremely tight bends. Especially when the tight bend is in the $t$ variable very near zero, stale tangents can be disasterous.





