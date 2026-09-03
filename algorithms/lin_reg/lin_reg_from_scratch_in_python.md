Linear Regression from Scratch, in Python
================

# Introduction

**What the algorithm does?** You give it points and, for each one, a
number you want to predict. It draws the straight line (or flat plane,
or hyperplane) that passes as close as possible to all of them. Labels
this time, not just geometry.

**What it minimizes? `MSE`.** Also called *mean squared error*: for every
point, take the **squared** gap between what you predicted and what
actually happened, and average them all. Small MSE = the line passes
close to the data. Unlike K-Means, finding the parameters with the
globally smallest MSE is **not** hard — the loss is a bowl (convex), so
there is exactly one bottom and calculus can hand it to you in closed
form. There are two routes to it:

1.  **The normal equation** — set the derivative to zero, solve the
    linear system, done. Exact, one shot, no iterations.
2.  **Gradient descent** — start anywhere, repeatedly step downhill:
    - **Predict** with the current parameters.
    - **Measure** how wrong you are.
    - **Differentiate** the loss to get the direction of steepest ascent.
    - **Step** the opposite way by a small amount `lr`.
    - **Repeat**; **stop** when the loss stops improving.

Every step of gradient descent lowers the loss (provided `lr` is small
enough), and the loss can’t go below its minimum — so the loop must
settle. Because the bowl has a single bottom, it settles on the *best*
possible answer. That is the headline difference from K-Means:
initialization is not a source of anxiety here.

> **Key Takeaway:**
>
> I.) Linear regression is convex, so unlike K-Means it has no local
> minima and no initialization lottery — every run finds the same
> answer. Gradient descent is not here to rescue you from a hard
> optimization problem; it is here for when the exact solution is too
> expensive to compute (too many features, too many rows) or when you
> want a template that generalizes to models that *aren’t* solvable in
> closed form.
>
> II.) Scale your features before **gradient descent**. The gradient in
> each direction is proportional to that feature’s magnitude, so a
> feature with range 0-1,000,000 produces a gradient a million times
> larger than one with range 18-80: a single `lr` cannot serve both, and
> you get divergence in one direction while crawling in the other. The
> **normal equation** does not care — it solves all directions at once.

# The worked example

Every step below runs on the same 6 points, so you can check numbers by
hand. One feature, so you can also plot it on paper.

``` python
import numpy as np

np.set_printoptions(precision=4, suppress=True)

X = np.array([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0],
    [6.0],
])

y = np.array([2.0, 4.1, 5.9, 8.2, 9.8, 12.0])

print(X.shape, y.shape)
#> (6, 1) (6,)
```

`X` is always `n x p`: one row per observation, one column per feature.
`y` is a flat vector of length `n`. Keeping that convention straight is
half the battle in every implementation below.

The parameters we are solving for are a weight vector `w` (length `p`)
and a single scalar intercept `b`. We start both at zero.

``` python
w_start = np.zeros(X.shape[1])
b_start = 0.0

print(w_start, b_start)
#> [0.] 0.0
```

Useful reference numbers, computed by hand once so you can check
everything else against them:

``` python
print("mean x :", X[:, 0].mean())
#> mean x : 3.5
print("mean y :", y.mean())
#> mean y : 7.0
```

# Step 0 — the loss function

Linear regression uses **squared error**: subtract prediction from
truth, square, sum, divide by `n`.

$$
\text{MSE}(w, b) = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2
\qquad \text{where} \qquad
\hat{y}_i = b + \sum_{d=1}^{p} w_d \, x_{id}
$$

Example, the guess `w = 2, b = 0`: predictions are
`2, 4, 6, 8, 10, 12` → errors `0.0, 0.1, -0.1, 0.2, -0.2, 0.0` → squares
`0, 0.01, 0.01, 0.04, 0.04, 0` → sum `0.1` → MSE `0.1 / 6 = 0.01667`.

**Why squared, and why this matters later:** three reasons that all
point the same way.

- It is **differentiable everywhere**, unlike absolute error which has a
  kink at zero. That is what makes gradients and the closed form
  possible at all.
- It is **convex in `w` and `b`**, so the derivative equals zero at
  exactly one place — the global minimum, not a local one.
- Setting the derivative to zero produces a system of *linear*
  equations, which is why Step 8 can solve it outright.

So “squared error” and “there is a formula for the answer” are a
*matched pair*, not two free choices. Swap in absolute error and you
lose the closed form immediately.

**Practical consequence:** squaring also means an error of 10 hurts 100
times as much as an error of 1, so a single outlier can drag the whole
line toward itself. Squared error buys you solvability and pays for it
with sensitivity to outliers.

``` python
def squared_error(y_true, y_pred):
    total = 0.0
    for i in range(len(y_true)):
        total += (y_true[i] - y_pred[i]) ** 2
    return total

# check: the guess w = 2, b = 0 -> 0.1
print(round(squared_error(y, 2.0 * X[:, 0]), 6))
#> 0.1
```

One more reference point. The dumbest possible model — ignore `X`
entirely and always predict the mean of `y` — gives the number every
other model has to beat. It comes back in Step 9.

``` python
print(round(squared_error(y, np.full(6, y.mean())), 4))
#> 68.9
```

# Step 1 — `predict`: parameters to predictions

A double loop turns `X`, `w` and `b` into an `n`-vector: for row `i`,
start at the intercept and add each feature times its weight.

**→ feeds Step 2**, which only needs the gap between these numbers and
`y`.

``` python
def predict(X, w, b):
    n, p = X.shape
    y_hat = np.zeros(n)

    for i in range(n):                 # each observation
        total = b                      # start at the intercept
        for d in range(p):             # each feature
            total += w[d] * X[i, d]
        y_hat[i] = total

    return y_hat


print(predict(X, w_start, b_start))
#> [0. 0. 0. 0. 0. 0.]

print(predict(X, np.array([2.0]), 0.0))
#> [ 2.  4.  6.  8. 10. 12.]
```

With `w = 0` and `b = 0` every prediction is `0` — the model has
learned nothing yet, which is exactly the starting state we want. With
`w = 2` it reproduces `2x`, which is very nearly the pattern in `y`.

# Step 2 — `compute_loss`: how wrong are we right now

Predict, subtract, square, average. This is the scoreboard: the single
number every later step exists to push down.

**← takes** the predictions from Step 1. **→ feeds** the convergence
check in Step 7.

``` python
def compute_loss(X, y, w, b):
    y_hat = predict(X, w, b)
    n = len(y)
    total = 0.0

    for i in range(n):
        total += (y[i] - y_hat[i]) ** 2

    return total / n


print(round(compute_loss(X, y, w_start, b_start), 6))
#> 60.483333

print(round(compute_loss(X, y, np.array([2.0]), 0.0), 6))
#> 0.016667
```

`60.48` at the origin, `0.0167` at the eyeballed guess — a factor of
3600. Note this is the **mean**, not the sum: dividing by `n` keeps the
number comparable across datasets of different sizes and keeps the
gradient from growing with the sample.

# Step 3 — `compute_gradients`: which way is downhill

Differentiate the MSE with respect to each parameter. Writing
`r_i = y_i - ŷ_i` for the residual:

$$
\frac{\partial \text{MSE}}{\partial w_d} = -\frac{2}{n} \sum_{i=1}^{n} x_{id} \, r_i
\qquad\qquad
\frac{\partial \text{MSE}}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n} r_i
$$

Read them out loud and they stop being algebra. The gradient for `b` is
just the **average residual**, times `-2`: if you are under-predicting
on average, push the intercept up. The gradient for `w_d` is the
**correlation between feature `d` and the residual**: if the points where
feature `d` is large are exactly the points you are under-predicting,
increase that weight.

The `-2` is a constant factor. It does not change the direction of the
step, only its length, which `lr` was going to rescale anyway — some
textbooks divide the loss by `2n` purely to cancel it.

**← takes** the predictions from Step 1. **→ feeds Step 4**, which
subtracts a multiple of these numbers.

``` python
def compute_gradients(X, y, w, b):
    n, p = X.shape
    y_hat = predict(X, w, b)

    residuals = np.zeros(n)
    for i in range(n):
        residuals[i] = y[i] - y_hat[i]

    grad_w = np.zeros(p)
    for d in range(p):                        # each feature
        total = 0.0
        for i in range(n):                    # correlate it with the residuals
            total += X[i, d] * residuals[i]
        grad_w[d] = -2.0 * total / n

    grad_b = -2.0 * residuals.sum() / n

    return grad_w, grad_b


grad_w, grad_b = compute_gradients(X, y, w_start, b_start)
print(np.round(grad_w, 4), round(grad_b, 4))
#> [-60.5667] -14.0
```

Hand-check both. At `w = 0, b = 0` every prediction is `0`, so every
residual *is* `y_i`. Then `grad_b = -2/6 * 42.0 = -14.0`, and
`grad_w = -2/6 * (1*2.0 + 2*4.1 + ... + 6*12.0) = -2/6 * 181.7 =
-60.5667`. Both are negative, meaning the loss falls if we *increase*
both parameters — which is obviously right, since the true line has a
positive slope and we are sitting at zero.

The gradient at the optimum is the definition of “nothing left to do”:

``` python
grad_w_opt, grad_b_opt = compute_gradients(X, y, np.array([1.982857142857143]), 0.06)
print(np.round(grad_w_opt, 12), round(grad_b_opt, 12))
#> [0.] -0.0
```

# Step 4 — `update_params`: take one step downhill

Subtract `lr` times the gradient. The gradient points uphill, so
subtracting walks down.

**Corner case — `lr` too large.** The gradient is only trustworthy
*locally*. Multiply it by too big a number and you leap past the bottom
of the bowl and land higher than you started; do that repeatedly and the
parameters explode to infinity. Step 6 shows it happening.

**← takes** the gradients from Step 3. **→ feeds Step 1 again** on the
next iteration.

``` python
def update_params(w, b, grad_w, grad_b, lr):
    p = len(w)
    new_w = np.zeros(p)

    for d in range(p):
        new_w[d] = w[d] - lr * grad_w[d]

    new_b = b - lr * grad_b

    return new_w, new_b


w_1, b_1 = update_params(w_start, b_start, grad_w, grad_b, lr=0.01)
print(np.round(w_1, 4), round(b_1, 4))
#> [0.6057] 0.14

print(round(compute_loss(X, y, w_1, b_1), 4))
#> 28.0169
```

`w` moved from `0` to `0.01 * 60.5667 = 0.6057`, `b` from `0` to
`0.01 * 14.0 = 0.14`, and the loss dropped from `60.48` to `28.02` in a
single step. Note that `w` moved more than four times as far as `b` —
not because it matters more, but because its gradient was four times
larger, which is entirely an artefact of `x` running from 1 to 6 rather
than being centred. That asymmetry is the seed of the scaling problem in
the Key Takeaway.

# Step 5 — one full iteration, by hand

Steps 1→2→3→4 chained once, then again, so you can watch the numbers
move before hiding them inside a loop.

``` python
w_iter, b_iter = w_start.copy(), b_start

for iteration in range(1, 4):
    g_w, g_b = compute_gradients(X, y, w_iter, b_iter)      # step 1 + step 3
    w_iter, b_iter = update_params(w_iter, b_iter, g_w, g_b, lr=0.01)   # step 4
    loss_i = compute_loss(X, y, w_iter, b_iter)             # step 2

    print(f"iteration {iteration}")
    print(f"  grad_w  : {g_w[0]:8.4f}   grad_b : {g_b:8.4f}")
    print(f"  w       : {w_iter[0]:8.4f}   b      : {b_iter:8.4f}")
    print(f"  loss    : {loss_i:8.4f}")
#> iteration 1
#>   grad_w  : -60.5667   grad_b : -14.0000
#>   w       :   0.6057   b      :   0.1400
#>   loss    :  28.0169
#> iteration 2
#>   grad_w  : -41.2148   grad_b :  -9.4803
#>   w       :   1.0178   b      :   0.2348
#>   loss    :  12.9903
#> iteration 3
#>   grad_w  : -28.0493   grad_b :  -6.4057
#>   w       :   1.2983   b      :   0.2989
#>   loss    :   6.0355
```

Loss `60.48 → 28.02 → 12.99 → 6.04`. Two things to notice.

First, the gradients **shrink** every iteration (`-60.6 → -41.2 →
-28.0`). Closer to the bottom, the bowl is flatter, so the same `lr`
produces a smaller step. Gradient descent slows itself down
automatically as it arrives — you do not have to schedule that.

Second, it is *not* converged. Contrast this with the K-Means
walkthrough, where iterations 2 and 3 were byte-identical to iteration 1.
K-Means has discrete assignments that snap into place and then stop;
gradient descent approaches its answer asymptotically and never exactly
arrives. That is why the stopping rule in Step 7 is a tolerance rather
than an equality test.
