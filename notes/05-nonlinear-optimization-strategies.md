# Nonlinear Optimization Strategies

## OUTLINE
• Nonlinear optimization in predictive

learning

• Overview of ANN
• Stochastic approximation (gradient

descent)
Iterative methods
•
• Greedy optimization
• Summary and discussion

## Optimization

• Optimization is concerned with the problem of

determining extreme values (maxima or minima) of a
function on a given domain.

• Let f(x) be a real-valued function of real variables x1, x2,

…, xm,
If x* minimizes unconstrained function f(x), then the
gradient of f(x) evaluated at x* is zero

• That is, x* is a solution of the system of equations

## Optimization

• The point where is called a stationary

or critical point;

• It can be a (local) minimum, a maximum, or a

saddle point of f(x)

The saddle point is a
local minimum along
the line x2 = -x1 and a
local maximum in the
orthogonal direction.

## Optimization

A critical point x* can be further checked for optimality by
considering the Hessian matrix of second partial
derivatives:

evaluated at x*. At a critical point x* where the gradient is
zero:

## Optimization

Second Derivative Test for Saddle Points
Let f(x, y) is a function in two variables for which the first
and second-order partial derivatives are continuous on
some disk that contains the point (a, b).
•If fx(a, b) = 0 and fy(a, b) = 0,

we define D = fxx(a, b) fyy (a, b) – [fxy(a, b)]
- If D > 0 and fxx(a, b) > 0, f has a local minimum at (a, b)
- If D > 0 and fxx(a, b) < 0, has a local maximum at (a, b)
- If D < 0, f has a saddle point (a, b)
- If D = 0, the test fails.

then:

2

[Saddle Points](https://byjus.com/maths/saddle-points/)

## Optimization

• With nonlinear optimization, there is always a possibility

of several local minima and saddle points.

• This has two important implications:

1. An optimization algorithm can find, at best, only a local

minimum.

2. The local minimum found by an algorithm is likely to be close to

an initial point x0.

The chances for obtaining globally optimal solution can be
improved (but not assured) by brute-force computational
techniques.

EX: Restarting optimization with many (randomized) initial
points and/or using simulated annealing to escape from
local minima.

## Optimization in predictive learning

• Optimization is used in learning

methods for parameter estimation.

• Recall implementation of SRM:
- fix complexity (VC-dimension)
- minimize empirical risk
Two related issues:
- parameterization (of possible models)
- optimization method

•

• Many learning methods use dictionary

parameterization

• Optimization methods vary

$$f_m(x,w,V) = \sum_{i=0}^{m} w_i g(x, v_i)$$

## Nonlinear Optimization

•

The ERM approach

where the model

Two factors contributing to nonlinear optimization

•

•

nonlinear basis functions

non-convex loss function L
Examples

convex loss: squared error, least-modulus
non-convex loss: 0/1 loss for classification

$$R_{emp}(V,W) = \frac{1}{n} \sum_{i=1}^{n} L(y_i, f(x_i, V, W))$$
$$f(x,V,W) = \sum_{j=1}^{m} w_j g(x, v_j)$$
$g(x, v_j)$

## Nonlinear Optimization - convex

[The-convex-optimization-function-Friedman-2002](https://www.researchgate.net/figure/The-convex-optimization-function-Friedman-2002_fig4_365210227)

## Unconstrained Minimization
• ERM or SRM (with squared loss) lead to

unconstrained convex minimization
• Minimization of a real-valued function of

many input variables
→ Optimization theory

• We discuss only popular optimization

strategies developed in statistics, machine
learning and neural networks

• These optimization methods have been
introduced in various fields and usually
use specialized terminology

## OUTLINE
• Nonlinear optimization in predictive

learning

• Overview of ANN
• Stochastic approximation (gradient

descent)
Iterative methods
•
• Greedy optimization
• Summary and discussion

## Overview of ANN’s

• Huge interest in understanding the

nature and mechanism of biological/
human learning

• Biologists + psychologists do not adopt
classical parametric statistical learning,
because:
- parametric modeling is not biologically
plausible
- biological info processing is clearly
different from algorithmic models of
computation

## Overview of ANN’s (Conti.)

• Mid 1980’s: growing interest in applying
biologically inspired computational
models to:

•

- developing computer models (of

human brain)

- various engineering applications
•
• → New field Artificial Neural Networks

(~1986 – 1987)

• ANN’s represent nonlinear estimators

implementing the ERM approach (usually
squared-loss function)

## History of ANN

McCulloch-Pitts neuron
Hebbian learning
Rosenblatt (perceptron), Widrow

1943
1949
1960’s
60’s-70’s dominance of ‘hard’ AI
1980’s

resurgence of interest (PDP group,
MLP etc.)
connection to statistics/VC-theory
1990’s
2000’s mature field/ lots of fragmentation
renewed interest ~ Deep Learning
2010’s

## Deep Learning
• New marketing term or smth different?

- several successful applications
- interest from the media, industry etc.
- very limited theoretical understanding

For critical & amusing discussion see:
Article in IEEE Spectrum on Big Data
[Machine-Learning-Maestro-Michael-Jordan-on-the-Delusions-of-Big-Data-and-Other-Huge-Engineering-Efforts](http://spectrum.ieee.org/robotics/artificial-intelligence/machinelearning-maestro-michael-jordan-on-the-delusions-of-big-data-and-other-huge-engineering-efforts)

And follow-up communications:

[Yann LeCun's post](https://www.facebook.com/yann.lecun/posts/10152348155137143)

[Big-Data-Hype-the-Media-and-Other-Provocative-Words-to-Put-in-a-Title](https://amplab.cs.berkeley.edu/2014/10/22/big-data-hype-the-media-and-other-provocative-words-to-put-in-a-title/)

## Neural vs Algorithmic computation
Biological systems do not use principles of
digital circuits

•

Digital
1~10
Connectivity
digital
Signal
synchronous
Timing
feedforward
Signal propag.
no
Redundancy
no
Parallel proc.
Learning
no
Noise tolerance no

Biological
~10,000
analog
asynchronous
feedback
yes
yes
yes
yes

## Neural vs Algorithmic computation
• Computers excel at algorithmic tasks (well-

•

posed mathematical problems)
Biological systems are superior to digital
systems for ill-posed problems with noisy data
Example: object recognition [Hopfield, 1987]

•
PIGEON: ~ 10^9 neurons, cycle time ~ 0.1 sec,

each neuron sends 2 bits to ~ 1K other neurons
→ 2x10^13 bit operations per sec

OLD PC: ~ 10^7 gates, cycle time 10^-7, connectivity=2

→ 10x10^14 bit operations per sec

Both have similar raw processing capability, but pigeons

are better at recognition tasks

## Goals of ANN’s

• Develop models of computation inspired by

biological systems
Study computational capabilities of networks
of interconnected neurons
Apply these models to real-life applications

•

•

Learning in NNs = modification (adaptation) of

synaptic connections (weights) in response to
external inputs

## [Perceptron](https://en.wikipedia.org/wiki/Perceptron)

## Neural terminology and artificial neurons

Some general descriptions of ANN’s:
[Neural network](http://en.wikipedia.org/wiki/Neural_network)
• McCulloch-Pitts neuron (1943)

•

Threshold (indicator) function of weighted sum of inputs

## Perceptron

The perceptron is an algorithm for learning a binary
classifier called a threshold function: a function that maps
its input x (a real-valued vector) to an output value

θ is the heaviside step-function

[Multi-layer-neural-networks-with-sigmoid-function-deep-learning-for-rookies-2](https://towardsdatascience.com/multi-layer-neural-networks-with-sigmoid-function-deep-learning-for-rookies-2-bf464f09eb7f)

## Perceptron

[Perceptron](https://en.wikipedia.org/wiki/Perceptron)

## Single layer to multi-layer

Step function has no useful derivative (its derivative is 0 everywhere
or undefined at the 0 point on x-axis). It doesn’t work for
backpropagation.
Hence, we need a better activation function!

[Multi-layer-neural-networks-with-sigmoid-function-deep-learning-for-rookies-2](https://towardsdatascience.com/multi-layer-neural-networks-with-sigmoid-function-deep-learning-for-rookies-2-bf464f09eb7f)

## Multilayer Perceptron (MLP) network

• Basis functions of the form

i.e. sigmoid aka logistic function

- commonly used in artificial neural networks
- combination of sigmoids ~ universal approximator

$$g(t) = g(v_i \cdot x + b_i)$$
$$s(t) = \frac{1}{1 + e^{-t}}$$

## [Multi-layer-neural-networks-with-sigmoid-function-deep-learning-for-rookies-2](https://towardsdatascience.com/multi-layer-neural-networks-with-sigmoid-function-deep-learning-for-rookies-2-bf464f09eb7f)

## Radial basis function

In mathematics a radial basis function (RBF)
is a real-valued function whose value
depends only on the distance between the
input and some fixed point, either the origin
or some other fixed point c called a center,

[Radial basis function](https://en.wikipedia.org/wiki/Radial_basis_function)

$$g(t) = g(|x - v_i|)$$

## RBF Networks

•

Basis functions of the form
i.e. Radial Basis Function(RBF)

- RBF adaptive parameters: center, width
- commonly used in artificial neural networks
- combination of RBF’s ~ universal approximator

$$g(t) = g(|x - v_i|)$$
$$g(t) = e^{-\frac{t^2}{2\alpha^2}}$$
$$g(t) = \frac{t^2}{t^2+b^2}$$
$g(t) = t$

## RBF Networks

[Radial basis function network](https://en.wikipedia.org/wiki/Radial_basis_function_network)

## MLP vs. RBF

• MLPs > RBFs
When the underlying characteristics feature of data is embedded deep
inside very high dimensional sets for example, in image recognition etc.
• RBFs > MLPs
When low dimensional data where deep feature extraction is not required
and results are directly correlated with the components of input vector.

[Radial-basis-functions-neural-networks](https://www.madrasresearch.org/post/radial-basis-functions-neural-networks)

## OUTLINE
• Nonlinear optimization in predictive

learning

• Overview of ANN
• Stochastic approximation (gradient

descent)
Iterative methods
•
• Greedy optimization
• Summary and discussion

## Sequential estimation of model parameters

•

Batch vs on-line (iterative) learning
- Algorithmic (statistical) approaches ~ batch
- Neural-network inspired methods ~ on-line
BUT the only difference is on the implementation level (so

both types of methods should yield similar generalization)

• Recall ERM inductive principle (for regression):

•

Assume dictionary parameterization with fixed basis fcts

$$R_{emp}(w) = \frac{1}{n} \sum_{i=1}^{n} L(y_i, f(x_i, w)) = \frac{1}{n} \sum_{i=1}^{n} (y_i - f(x_i, w))^2$$
$$\hat{y} = f(x,w) = \sum_{j=1}^{m} w_j g_j(x)$$

## Sequential (on-line) least squares minimization

Training pairs presented sequentially

•
• On-line update equations for minimizing

empirical risk (MSE) wrt parameters w are:

(gradient descent learning)

where the gradient is computed via the chain rule:

the learning rate
(decreasing with k)

is a small positive value

$(x(k), y(k))$
$$w_{k+1} = w_k - \gamma_k \frac{\partial L(y(k), f(x(k), w_k))}{\partial w}$$
$$\frac{\partial L(x,y,w)}{\partial w_j} = \frac{\partial L}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial w_j} = 2(\hat{y} - y) g_j(x)$$
$\gamma_k$

## On-line least-squares minimization algorithm

• Known as delta-rule (Widrow and Hoff, 1960):
Given initial parameter estimates w(0), update

parameters during each presentation of k-th
training sample x(k),y(k)
Step 1: forward pass computation

- estimated output

Step 2: backward pass computation

- error term (delta)

•

•

$$z_j(k) = g_j(x(k)), j=1,...,m$$
$$\hat{y}(k) = \sum_{j=1}^{m} w_j(k) z_j(k)$$
$$\delta(k) = \hat{y}(k) - y(k)$$
$$w_j(k+1) = w_j(k) - \gamma_k \delta(k) z_j(k), j=1,...,m$$

## Neural network interpretation of delta rule

•

Forward pass

Backward pass

•

“Biological learning”
- parallel+ distributed comp.
- can be extended to

multiple-layer networks

$$\Delta w_j(k) = \gamma_k \delta(k) z_j(k)$$
$$w_j(k+1) = w_j(k) + \Delta w_j(k)$$

## Backpropagation Training of
MLP Networks

Consider a set of approximating functions

The risk functional is

The stochastic approximation procedure for minimizing
this risk with respect to the parameters V and w is

## Backpropagation Training of
MLP Networks

The loss L is

The gradient of the loss L can be computed via the chain
rule of derivatives if the approximating function is
decomposed as

## Backpropagation Training of
MLP Networks

Based on the chain rule, the relevant gradients are

We can calculate these partial derivatives using functions
in the previous slide.

## Backpropagation Training of
MLP Networks

• With these gradients and the

stochastic approximation updating
equations, it is now possible to
construct a computational
procedure to find the local
minimum of the empirical risk.

## Backpropagation training:
(a) forward pass;
(b) backward pass.

## •

•

Theoretical basis for on-line learning

Standard inductive learning: given training
data find the model providing min of
prediction risk

Stochastic Approximation guarantees
minimization of risk (asymptotically):

under general conditions
on the learning rate:

$$R(\omega) = \int L(z, \omega) p(z) dz$$
$z_1, ..., z_n$
$$\omega_{k+1} = \omega_k - \gamma_k \nabla_{\omega} L(z_k, \omega_k)$$
$\lim_{k \to \infty} \gamma_k = 0$
$\sum_{k=1}^{\infty} \gamma_k = \infty$
$\sum_{k=1}^{\infty} \gamma_k^2 < \infty$

## •

•

•

Practical issues for on-line learning

Given finite training set (n samples):
this set is presented to a sequential learning algorithm
many times.
1.

Epoch is the complete passing through of all the datasets
exactly at once.
Batch is the dataset that has been divided into smaller parts to
be fed into the algorithm.

2.

Learning rate schedule: initially set large, then slowly
decreasing with k (iteration number). Typically ’good’
learning rate schedules are data-dependent.
Stopping conditions:
(1) monitor the gradient (i.e., stop when the gradient
falls below some small threshold)
(2) early stopping can be used for complexity control

[Machine learning tutorial](https://www.simplilearn.com/tutorials/machine-learning-tutorial)

$z_1, ..., z_n$

## OUTLINE
• Nonlinear optimization in predictive

learning

• Overview of ANN
• Stochastic approximation (gradient

descent)
Iterative methods
•
• Greedy optimization
• Summary and discussion

## • Dictionary representation with fixed m :

Iterative strategy for ERM (nonlinear optimization)

•

Step 1: for current estimates of update
Step 2: for current estimates of update

•
Iterate Steps (1) and (2) above

Note: - this idea can be implemented for different problems:

e.g., unsupervised learning, supervised learning

Specific update rules depend on the type of problem & loss fct.

$$f(x,V,w) = \sum_{i=0}^{m} w_i g(x,v_i)$$
$\hat{v}_i$
$\hat{w}_i$

## Iterative Methods

•

Implement iterative parameter
estimation.

• This leads to a generic parameter

estimation scheme, where the two steps
(expectation and maximization) are
iterated until some convergence criterion
is met.

• Expectation-Maximization (EM)
developed/used in statistics.

## EM Methods for Density
Estimation

Motivating Example:
•Have two coins: Coin1 and Coin2
1. Select a coin at random and flip that one coin m times.
2. Repeat this process n times.
•We have

•Note that all the X’s are independent and, in particular

[EM algorithm](https://www.colorado.edu/amath/sites/default/files/attached-files/em_algorithm.pdf)

## EM Methods for Density
Estimation

• We can write out the joint pdf of all nm+n random

variables and formally come up with MLEs for p1 and p2.

• Call these MLEs p1 and p2. They will turn out as

expected:

• Now suppose that the Yi are not observed but we still
want MLEs for p1 and p2. The data set now consists of
only the X’s and is “incomplete”.

• The goal of the EM Algorithm is to find MLEs for p1 and

p2 in this case.

[EM algorithm](https://www.colorado.edu/amath/sites/default/files/attached-files/em_algorithm.pdf)

## EM Methods for Density
Estimation

• Let X be observed data, generated by some distribution depending

on some parameters.

• Let Y be some “hidden” or “unobserved data” depending on some

parameters.

• Let Z=(X,Y) represent the “complete” dataset.
• Assume we can write the pdf for Z as (depends on some parameter

θ)

• We have the complete likelihood function:

• We have the incomplete likelihood function:

[EM algorithm](https://www.colorado.edu/amath/sites/default/files/attached-files/em_algorithm.pdf)

## EM Methods for Density
Estimation
• The EM Algorithm is a numerical iterative for finding an
MLE of θ. The rough idea is to start with an initial guess
for θ and to use this and the observed data X to
“complete” the dataset by using X and the guessed θ to
postulate a value for Y, at which point we can then find
an MLE for θ in the usual way.

• We will use an initial guess for θ and postulate an entire
distribution for Y, ultimately averaging out the unknown
Y.

[EM algorithm](https://www.colorado.edu/amath/sites/default/files/attached-files/em_algorithm.pdf)

## EM Methods for Density Estimation

The EM Algorithm is iterated until the estimate for θ stops changing.

[EM algorithm](https://www.colorado.edu/amath/sites/default/files/attached-files/em_algorithm.pdf)

## EM Methods for Density
Estimation

a) Two hundred data points
drawn from a doughnut
distribution.

b) Initial configuration of five

Gaussian mixtures.
c) Configuration after 20

iterations of the EM algorithm

## OUTLINE
• Nonlinear optimization in predictive

learning

• Overview of ANN
• Stochastic approximation (gradient

descent)
Iterative methods
•
• Greedy optimization
• Summary and discussion

## Greedy Optimization Methods

• Minimization of empirical risk (regression problems)

where the model

•

Greedy Optimization Strategy
basis functions are estimated sequentially, one at a time,

i.e., the training data is represented as
structure (model fit) + noise (residual):

(1) DATA = (model) FIT 1 + RESIDUAL 1
(2) RESIDUAL 1 = FIT 2 + RESIDUAL 2

and so on. The final model for the data will be

MODEL = FIT 1 + FIT 2 + ....

•

Advantages: computational speed, interpretability

$$R_{emp}(V,W) = \frac{1}{n} \sum_{i=1}^{n} (y_i - f(x_i, V, W))^2$$
$$f(x,V,W) = \sum_{j=1}^{m} w_j g(x, v_j)$$

## Classification and Regression Trees (CART)

• Minimization of empirical risk (squared error)
via partitioning of the input space into regions

•

Example of CART partitioning for a function of 2 inputs

$$f(x) = \sum_{j=1}^{m} w_j I(x \in R_j)$$

## Growing CART

• Recursive partitioning for estimating regions

•

•
•

•

(the whole input domain)
and

(via binary splitting)
Initial Model ~ Region
is divided into two regions
A split is defined by one of the inputs(k) and split point s
Optimal values of (k, s) chosen so that splitting a region
into two daughter regions minimizes empirical risk
Issues:
- efficient implementation (selection of opt. split)
- optimal tree size ~ model selection(complexity control)

$R_0$
$R_1$
$R_2$

## Growing CART

• Advantages:
1. Results are simplistic.
2. Classification and regression trees are Nonparametric and

Nonlinear.

3. Classification and regression trees implicitly perform feature

selection.

4. Outliers have no meaningful effect on CART.
5. It requires minimal supervision and produces easy-to-

understand models.

• Limitations:
1. Overfitting.
2. High Variance.
3. Low bias.
4. The tree structure may be unstable.

[CART classification and regression tree in machine learning](https://www.geeksforgeeks.org/cart-classification-and-regression-tree-in-machine-learning/)

## CART Example
• CART model for estimating Systolic Blood

Pressure (SBP) as a function of Age and
Obesity in a population of males in S. Africa

## CART model selection

• Model selection strategy

(1) Grow a large tree (subject to min leaf node size)
(2) Tree pruning by selectively merging tree nodes

•

The final model ~ minimizes penalized risk

where empirical risk ~ MSE

number of leaf nodes ~
regularization parameter ~

• Note: larger → smaller trees
•

In practice: often user-defined (splitmin in Matlab)

$$R_{pen}(\omega, \lambda) = R_{emp}(\omega) + \lambda T$$
$T$
$\lambda$

## •

•

Pitfalls of greedy optimization

Greedy binary splitting strategy may yield:
- sub-optimal solutions (regions )
- solutions very sensitive to random samples (especially
with correlated input variables)

Counterexample for CART:
estimate function f(a,b,c):
- which variable for the first split?
Choose a → 3 errors
Choose b → 4 errors
Choose c → 4 errors

y a b c
0
0
0
0
1
1
1
1
0
1
0
1
1
0
1
0

0
1
0
1
0
1
1
1

0
0
0
0
1
1
1
1

$R_i$

## Counter example (cont’d)

(a) Suboptimal tree by CART

(b) Optimal binary tree

## Gini score

1-(2/5)^2-(3/5)^2= 1-4/25-9/25=12/25=0.48

pi is the probability of an object being
classified to a particular class.

5/14*0.48+5/14*0.48=0.343

[Gini impurity](https://www.learndatasci.com/glossary/gini-impurity/)

## Backfitting Algorithm
• Consider regression estimation of a function of

two variables of the form
from training data
For example
Backfitting method:

(1) estimate for fixed
(2) estimate for fixed
iterate above two steps

•

Estimation via minimization of empirical risk

$$R_{emp}(g_1, g_2) = \frac{1}{n} \sum_{i=1}^{n} (y_i - g_1(x_{i1}) - g_2(x_{i2}))^2$$
$$y = g_1(x_1) + g_2(x_2) + noise$$
$$g_1(x_1) = 2sin(2\pi x_1), g_2(x_2) = 2x_2^2$$
$(x_{i1}, x_{i2}, y_i)$
$x \in [0,1]^2$
$g_1(x_1)$
$g_2$
$g_2(x_2)$
$g_1$

## •

•

Backfitting Algorithm(cont’d)
Estimation of via minimization of MSE

This is a univariate regression problem of
estimating from n data points
where

• Can be estimated by smoothing (kNN regression)
Estimation of (second iteration) proceeds
•
in a similar manner, via minimization of

where

•

Backfitting ~gradient descent in the function space

$$\min_{g_1} R_{emp}(g_1) = \frac{1}{n} \sum_{i=1}^{n} (r_{i1} - g_1(x_{i1}))^2$$
$$r_{i1} = y_i - g_2(x_{i2})$$
$g_1(x_1)$
$(r_{i1}, x_{i1})$
$g_2(x_2)$
$$R_{emp}(g_2) = \frac{1}{n} \sum_{i=1}^{n} (r_{i2} - g_2(x_{i2}))^2$$
$$r_{i2} = y_i - g_1(x_{i1})$$

## OUTLINE
• Nonlinear optimization in predictive

learning

• Overview of ANN
• Stochastic approximation (gradient

descent)
Iterative methods
•
• Greedy optimization
• Summary and discussion

## Summary
• Different interpretation of optimization
• Consider dictionary parameterization

• VC-theory: implementation of SRM
• Gradient descent + iterative optimization

- SRM structure is specified a priori
- selection of m is separate from ERM

• Greedy optimization strategy

- no a priori specification of a structure
- model selection is a part of optimization

$$f_m(x,w,V) = \sum_{i=0}^{m} w_i g(x, v_i)$$

## Summary (cont’d)
Interpretation of greedy optimization

•

• Statistical strategy for iterative data fitting

(1) Data = (model) Fit1 + Residual_1

(2) Residual_1 = (model) Fit2 + Residual_2
………..

→ Model = Fit1 + Fit2 + …

• This has superficial similarity to SRM

$$f_m(x,w,V) = \sum_{i=0}^{m} w_i g(x, v_i)$$
