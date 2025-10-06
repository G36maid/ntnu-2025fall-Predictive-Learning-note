# Regularization and Complexity Control

## OUTLINE
(following Cherkassky and Mulier, 2007 Chapter 3)
• Curse of Dimensionality
• Function approximation framework
• Penalization/ regularization inductive

principle

• Bias-variance trade-off
• Complexity Control
• Predictive learning vs. function

approximation

• Summary

## Curse of Dimensionality

• In the learning problem, the goal is to estimate
a function using a finite number of training
samples.

• Meaningful estimation is possible only for
sufficiently smooth functions, where the
function smoothness is measured with respect
to sampling density of the training data.
• For high-dimensional functions, it becomes

difficult to collect enough samples to attain this
high density.

• This problem is commonly referred to as the

‘‘curse of dimensionality.’’

## Curse of Dimensionality
• High-dimensional distribution, e.g.,

hypercube, if it could be visualized, would
look like a porcupine

•

Sample size needed for accurate fct estimation
grows exponentially with dimensionality d

## Four properties of high-dimensional
distributions

1. Sample sizes yielding the same density increase

exponentially with dimension.
Let us assume that for R1, a sample containing n data points is
considered a dense sample. To achieve the same density of points in
d dimensions, we need nd data points.

2. A large radius is needed to enclose a fraction of
the data points in a high-dimensional space.

Both gray regions enclose 10
percent of the samples

.32.46

## Four properties of high-dimensional
distributions

3. Almost every point is closer to an edge than to
another point.

Consider a situation where n data points are uniformly distributed in a
d-dimensional ball with unit radius. The median distance between the
center of the distribution (the origin) and the closest data point is

n = 10

n = 100

d = 10

0.763

0.609

d = 100

0.973

0.951

4. Almost every point is an outlier in its own
projection.

Conceptual illustration: To someone standing on the
end of a ‘‘quill’’ of the porcupine, facing the center of
the distribution, all the other data samples will appear
far away and clumped near the center.

## Problems of curse of dimensionality

• Difficult to make local estimates for high-
dimensional samples (from properties 1
and 2).

• Difficult to predict a response at a given

point because any point will on average be
closer to an edge than to the training data
point (from properties 3 and 4).

## Mathematical theorems contradict the
curse of dim??
The Curse: with finite samples learning in high-
dimensions is very hard/ impossible ?
BUT
Kolmogorov's theorem states that any continuous
function of d arguments can be represented as
superposition of univariate functions

•

•

→ no curse of dimensionality ???

## Mathematical theorems contradict the
curse of dim??

Explanation: difficulty of learning depends on the

complexity of target functions, but Kolmogorov’s
theorem does not quantify complexity.

We can conclude that
•

A function’s dimensionality is not a good measure
of its complexity.

• High-dimensional functions have the potential to
be more complex than low-dimensional functions.
There is a need to provide a characterization of a
function’s complexity that takes into account its
smoothness and dimensionality.

•

## OUTLINE
(following Cherkassky and Mulier, 2007 Chapter 3)
• Curse of Dimensionality
• Function approximation framework
• Penalization/ regularization inductive

principle

• Bias-variance trade-off
• Complexity Control
• Predictive learning vs. function

approximation

• Summary

## Function Approximation

• How to represent/approximate any (continuous)

target function via given class of basis
functions?

Weierstrass theorem
• Any continuous function on a compact set can
be uniformly approximated by a polynomial.
• In other words, for any such function f(x) and
any positive Ɛ, there exists a polynomial of
degree m, pm(x), such that

for every x.

## Universal approximation

• Any (continuous) function can be accurately

approximated by another function from a given
class (i.e., as in the Weierstrass theorem stated
above).

• Most universal approximators represent a linear
combination of basis functions (also known as
the dictionary method)

$$f(x, w) = \sum_{i=0}^{m} w_i g_i(x)$$

where gi are the basis functions,
are parameters.

## Universal approximation

• Algebraic polynomials

• Trigonometric polynomials

## Universal approximation

• Multilayer networks

• Local basis function networks

## Rate-of-convergence

• Relate the accuracy of approximation to the

properties of target function (its dimensionality d
and smoothness s)
• Rate-of-convergence

$O(m^{-s/d})$

where s = number of continuous derivatives
(~smoothness)

• For a given approximation error, the number of

parameters exponentially increases with d (for a
fixed measure of ‘‘complexity’’ s).

## Characterization of a function’s
complexity

1. Define the measure of complexity for a class of

target functions.
This class of functions should be very general, so that it
is likely to include most target functions in real-life
applications.

2. Specify a class of approximating functions of a

learning machine.
For example, choose a particular dictionary in
representation. This dictionary should have ‘‘the
universal approximation’’ property.
Flexibility of approximating functions is specified by the
number of basis functions m.

## Characterization of a function’s
complexity

3. Estimate the (best possible) rate of

convergence, defined as the accuracy of
approximating an arbitrary function in the class.
In other words, estimate how quickly the
approximation error of a method goes to zero
when the number of its parameters grows large.

It is of particular interest to see how the rate of
convergence depends on the dimensionality of
the class of functions.

## Some related issues

• How to measure function complexity?

•

•

Number of continuous derivatives s (smoothness).
Frequency content (i.e., max frequency)
Fundamental restriction:
good function approximation (in high dimension) is
possible only for very smooth functions
Implications for machine learning methods:
- good generalization not possible (for high-dim. data),
- need to constrain/control smoothness for multivariate
problems → penalization inductive principle.

## OUTLINE
• Curse of Dimensionality
• Function approximation framework
• Penalization/ regularization inductive

principle

• Bias-variance trade-off
• Complexity Control
• Predictive learning vs. function

approximation

• Summary

## Penalization Inductive Principle

• Specify a wide set of models
• Find a model minimizing Penalized Risk

$$R_{pen}(\omega) = R_{emp}(\omega) + \lambda \Phi(\omega)$$

~ non-negative penalty functional;
its larger values penalize complex functions.
~ regularization parameter controls the strength

of penalty relative to empirical risk (data term)
Model Selection problem: select so that the solution

found by minimizing provides minimum

expected risk aka prediction risk aka test error.
→ Need to estimate expected risk for each solution

$f(x, \omega, \lambda)$
$R_{pen}(\omega)$

## Penalization of linear models

• Linear function:

• Cost function:

• LASSO (L1 regularization):

• Ridge (L2 regularization):

[tutorial-lasso-ridge-regression](https://www.datacamp.com/tutorial/tutorial-lasso-ridge-regression)

## Penalization of linear models

• Which one can perform feature selection?

[tutorial-lasso-ridge-regression](https://www.datacamp.com/tutorial/tutorial-lasso-ridge-regression)

## Modeling Issues for Penalization

• Choice of admissible models

(1) continuous functions
(2) wide class of parametric functions
Type of penalty ~ prior knowledge

•

(1) nonparametric (smoothness constraints)
(2) parametric, e.g. ridge penalty, subset selection

• Method for minimizing
• Choice of ~ complexity control
→ Final model depends on all factors above BUT
different tradeoffs for small vs. large samples

$\Phi(f(x, \omega))$
$f(x, \omega)$
$R_{pen}(\omega)$

## OUTLINE
• Curse of Dimensionality
• Function approximation framework
• Penalization/ regularization inductive

principle

• Bias-variance trade-off
• Complexity Control
• Predictive learning vs. function

approximation

• Summary

## Bias-Variance Trade-Off
Statistical ‘explanation’ for model complexity control

Recall (a) Classification

(b) Regression

Imagine many training data sets of the same size

## Bias-Variance Trade-Off
• For regression problems with squared loss:
Consider MSE btwn an estimate and the target fct.
Note: MSE is averaged over many possible
training samples of the same size n

this MSE can be expressed as

where

$$mse(f(x, \omega)) = E[\int (g(x) - f(x, \omega))^2 p(x) dx]$$
$$mse(f(x, \omega)) = bias^2(f(x, \omega)) + var(f(x, \omega))$$
$$bias^2(f(x, \omega)) = \int (g(x) - E[f(x, \omega)])^2 p(x) dx$$
$$var(f(x, \omega)) = E[\int (f(x, \omega) - E[f(x, \omega)])^2 p(x) dx]$$

## Bias vs Varaince

• The bias error is an error from erroneous assumptions in
the learning algorithm. High bias can cause an algorithm
to miss the relevant relations between features and
target outputs (underfitting).

• The variance is an error from sensitivity to small

fluctuations in the training set. High variance may result
from an algorithm modeling the random noise in the
training data (overfitting).

[Wikipedia](https://en.wikipedia.org/)

$$bias^2(f(x, \omega)) = \int (g(x) - E[f(x, \omega)])^2 p(x) dx$$
$$var(f(x, \omega)) = E[\int (f(x, \omega) - E[f(x, \omega)])^2 p(x) dx]$$

## High bias, low variance

High bias, high variance

Low bias, low variance

Low bias, high variance

[Wikipedia](https://en.wikipedia.org/)

## •

Bias-Variance trade-off and Penalization
Parameter controls bias-variance trade-off:
- larger values → smaller variance/ larger bias
- smaller values → larger variance (the model
becomes more dependent on the training data)

$\lambda$

## Example of high bias (underfitting)

50 training samples (5 data sets)
Gaussian kernel width ~ 80%

-0.4-0.200.20.40.60.8100.20.40.60.81XY

## Example of high variance (overfitting)

Same training data (5 data sets)
Piecewise-linear regression fitting (10 components)

-0.500.511.500.20.40.60.81XY

## Summary: bias-variance trade-off

• The concept of complexity control
• Model estimation depends on two terms:

- data (empirical risk)
- penalization term (~ a priori knowledge)
• A particular set of models (approximating
functions) cannot be ‘best’ for all data sets

• Bias-variance formalism does not provide practical

mechanism for model selection

## OUTLINE

• Curse of Dimensionality
• Function approximation framework
• Penalization/ regularization inductive

principle

• Bias-variance trade-off
• Complexity Control
• Predictive learning vs. function

approximation

• Summary

## How to Control Model Complexity ?

• Two approaches: analytic and resampling
• Analytic criteria estimate prediction error as a function of

fitting error and model complexity
For regression problems:

Representative analytic criteria for regression
• Schwartz Criterion:

• Akaike’s FPE:

where p = DoF/n, n~sample size, DoF~degrees-of-freedom

$$R(\omega) \approx R_{emp}(\omega) \left( \frac{1 + \frac{DoF}{n}}{1 - \frac{DoF}{n}} \right)$$
$$r_p = \frac{1 + \sqrt{p}}{1 - \sqrt{p}}, p = \frac{DoF}{n}$$
$$r_p = \frac{1 + p}{1 - p}$$

## Analytical Model Selection

• Consider estimating the data using the set of polynomial

approximating functions of arbitrary degree

• For practical purposes, we will limit the polynomial

degree m ≤ 10.

## Resampling
• Split available data into 2 subsets:

Training + Validation
(1) Use training set for model estimation
(via data fitting)
(2) Use validation data to estimate the
prediction error of the model

• Change model complexity index and

repeat (1) and (2)

• Select the final model providing the
lowest (estimated) prediction error

BUT results are sensitive to data splitting

## K-fold cross-validation

1. Divide the training data Z into k (randomly selected)

disjoint subsets {Z1, Z2,…, Zk} of size n/k

2. For each ‘left-out’ validation set Zi :

- use remaining data to estimate the model
- estimate prediction error on Zi :

3. Estimate ave. prediction risk as

$y = f(x_i)$
$$R_{cv} = \frac{1}{k} \sum_{i=1}^{k} R_i$$
$$R_i = \frac{1}{n/k} \sum_{z_j \in Z_i} (y_j - f(x_j))^2$$

## K-fold cross-validation

• Consider 25 samples for cross-validation

## K-fold cross-validation

## Example of model selection(1)

• 25 samples are generated as

with x uniformly sampled in [0,1], and noise ~ N(0,1)

• Regression estimated using polynomials of degree m=1,2,…,10
• Polynomial degree m = 5 is chosen via 5-fold cross-validation.
The curve shows the polynomial model, along with training (* )
and validation (*) data points, for one partitioning.

m

1

2

3

4

5

6

7

8

9

Estimated R via
Cross validation

0.1340

0.1356

0.1452

0.1286

0.0699

0.1130

0.1892

0.3528

0.3596

10

0.4006

$y = sin^2(2\pi x) + \xi$

## Example of model selection(2)

• Same data set, but estimated using k-nn regression.
• Optimal value k = 7 chosen according to 5-fold cross-validation
model selection. The curve shows the k-nn model, along with
training (* ) and validation (*) data points, for one partitioning.

k

1

2

3

4

5

6

7

8

9

10

Estimated R via
Cross validation

0.1109

0.0926

0.0950

0.1035

0.1049

0.0874

0.0831

0.0954

0.1120

0.1227

## More on Resampling

• Leave-one-out (LOO) cross-validation
- extreme case of k-fold when k=n (# samples)
- efficient use of data, but requires n estimates

• Final (selected) model depends on:

- random data
- random partitioning of the data into K subsets (folds)
→ the same resampling procedure may yield different
model selection results

• Some applications may use non-random splitting of the

data into (training + validation)

• Model selection via resampling is based on estimated

prediction risk (error).

• Does this estimated error measure reflect true prediction

accuracy of the final model?

## Resampling for estimating true risk

• Prediction risk (test error) of a method can be

also estimated via resampling

• Partition the data into: Training/ validation/ test
• Test data should be never used for model

estimation

• Double resampling method:

- for complexity control
- for estimating prediction performance of a
method

• Estimation of prediction risk (test error) is critical

for comparison of different learning methods

## On resampling terminology

• Often confusing and inconsistent terminology

- Resampling for model selection:

• Double resampling for estimating test error and

model selection

## Example of model selection for k-NN
classifier via 6-fold x-validation: Ripley’s data.
Optimal decision boundary for k=14

-1.5-1-0.500.51-0.200.20.40.60.811.2

## Example of model selection for k-NN
classifier via 6-fold x-validation: Ripley’s data.
Optimal decision boundary for k=50

which one
is better?
k=14 or 50

-1.5-1-0.500.51-0.200.20.40.60.811.2

## Estimating test error of a method
For the same example (Ripley’s data) what is the true
test error of k-NN method ?
Use double resampling, i.e. 5-fold cross validation to
estimate test error, and 6-fold cross-validation to
estimate optimal k for each training fold:

k
20
9
1
12
7

Validation
11.76%
0%
17.65%
5.88%
17.65%
10.59%

Fold #
1
2
3
4
5
mean
Note: opt k-values are different; errors vary for each fold,
due to high variability of random partitioning of the data

Test error
14%
8%
10%
18%
14%
12.8%

•

•

•

## •

•

Estimating test error of a method
Another realization of double resampling, i.e. 5-fold
cross validation to estimate test error, and 6-fold cross-
validation to estimate optimal k for each training fold:

Fold #
1
2
3
4
5
mean

k
7
31
25
1
62

Validation
14.71%
8.82%
11.76%
14.71%
11.76%
12.35%

Test error
14%
14%
10%
18%
4%
12%

Note: predicted average test error (12.8%) is usually higher
than minimized validation error (10.6%) for model selection,
as shown in the first realization

## OUTLINE

• Curse of Dimensionality
• Function approximation framework
• Penalization/ regularization inductive

principle

• Bias-variance trade-off
• Complexity Control
• Predictive learning vs. function

approximation

• Summary

## Inductive Learning Setting

• Consider regression problem:
• Goal of function approximation (system identification):

• Goal of Predictive Learning:

Note: p(x) denotes unknown pdf for the input (x) values.

$y(t) = x(t) + \xi$
$$\min_{f} \int (g(x) - f(x,w))^2 dx$$
$$\min_{f} \int (y(x) - f(x,w))^2 p(x) dx$$

Generator of samplesLearning MachineSystemxyˆ y 

## Example from [Cherkassky & Mulier, 2007, Ch. 3]
Regression estimation: penalized polynomials of degree 15);
30 training samples / 30 validation samples (for tuning lambda)

Target function

Distribution of x
(training data)

## Typical Results/ comparisons for
Predictive setting: validation data ~ normal distribution
Function approximation: validation ~ uniform distribution in [0,1]

Dotted line ~ predictive setting / Dashed ~ fct approximation

## Conclusion
• The goal of predictive learning is different from

function approximation

• Function approximation ~ accurate estimation

everywhere in the input domain.

• Predictive learning ~ reduce the overall prediction
risk (focus on the data which is more likely to be
observed).

## Summary
• Regularization/ Penalization ~ provides good

math framework for predictive learning

• Important distinction:

predictive learning vs. system identification

• Bias-variance trade-off

• Complexity control and resampling

- analytic criteria (typically for regression only);
- resampling methods (for all types of problems)
