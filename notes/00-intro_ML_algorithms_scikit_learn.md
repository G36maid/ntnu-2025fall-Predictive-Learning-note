# 00-Introduction to ML Algorithms and Scikit-learn

## Instructor
Hsiang-Han Chen (陳翔瀚)

*(Please do not distribute without author's permission.)*

## Overview of Machine Learning

Machine learning can be broadly categorized into:

*   **Supervised Learning**
    *   Classification
    *   Regression
*   **Unsupervised Learning**
    *   Clustering
    *   Dimensionality Reduction
*   **Model Selection**
*   **Preprocessing**

## Supervised Learning

### Classification

Identifying which category an object belongs to.

*   **Applications:** Spam detection, image recognition.
*   **Algorithms:** Gradient boosting, nearest neighbors, random forest, logistic regression, etc.

### Regression

Predicting a continuous-valued attribute associated with an object.

*   **Applications:** Drug response, stock prices.
*   **Algorithms:** Gradient boosting, nearest neighbors, random forest, ridge, etc.

### Classification vs. Regression Resources

*   [Regression vs Classification in Machine Learning](https://www.simplilearn.com/regression-vs-classification-in-machine-learning-article)
*   [Regression vs. Classification: What's the difference?](https://www.springboard.com/blog/data-science/regression-vs-classification/)

## Linear Models

### Linear Models for Classification

For classification, linear models can be used. Further details can be found in this discussion:
*   [How can a linear model be used for classification?](https://stats.stackexchange.com/questions/22381/)

### Linear Models – Penalization

Penalization techniques are used to prevent overfitting in linear models:

*   **Ridge Model:** (L2 regularization)
*   **Lasso Model:** (L1 regularization)
*   **Elastic-Net:** (Combination of L1 and L2 regularization)

### Linear Models – Generalized Linear Models (GLM)

GLMs extend linear models in two ways:

1.  **Inverse Link Function:** Predicted values ($\hat{y}$) are linked to a linear combination of input variables ($X$) via an inverse link function $h$, as in $\hat{y} = h(X\omega)$.
    *   **Example:** With `link='log'`, the inverse link function becomes $\exp(X\omega)$.
2.  **Loss Function:** The squared loss function is replaced by the unit deviance ($d$) of a distribution in the exponential family (or a reproductive exponential dispersion model (EDM)).

### Generalized Linear Models – Specific EDMs and their Unit Deviance

(Further details on specific EDMs and their unit deviance would typically be provided here in a detailed note.)

### Logistic Regression

Despite its name, logistic regression is implemented as a linear model for classification in `scikit-learn` and ML nomenclature.

It is a special case of Generalized Linear Models with a Binomial/Bernoulli conditional distribution and a Logit link.

### Polynomial Regression

Extends linear models using basis functions.

*   **Example:** If $X$ is a feature, new features like $X^2$, $X^3$ can be added.
*   This is still considered a linear model by creating new features, e.g., $X_1 = X$, $X_2 = X^2$.

## Linear and Quadratic Discriminant Analysis

Both Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA) are derived from simple probabilistic models that model the class conditional distribution of the data.

Predictions are obtained using Bayes' rule for each training sample.

### Mathematical Formulation

For LDA and QDA, $P(x|y)$ is modeled as a multivariate Gaussian distribution with density:

$p(x|y=k) = \frac{1}{(2\pi)^{d/2} |\Sigma_k|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k)\right)$

Where $d$ is the number of features.

**QDA:** The log of the posterior is:

$\log P(y=k|x) = \log P(x|y=k) + \log P(y=k) + Cst$

Where $Cst$ corresponds to the denominator $P(x)$ and other constant terms. The predicted class maximizes this log-posterior.

**LDA:** A special case of QDA where the Gaussians for each class are assumed to share the same covariance matrix: $\Sigma_k = \Sigma$ for all $k$.

This reduces the log posterior. The log-posterior of LDA can also be written in a form that clearly shows LDA has a linear decision surface.

## Support Vector Machines (SVM)

SVM methods are used for classification, regression, and outlier detection.

### Advantages of SVMs:

*   Effective in high-dimensional spaces.
*   Still effective even when the number of dimensions is greater than the number of samples.
*   Uses a subset of training points (support vectors) in the decision function, making them memory efficient.
*   Various Kernel functions can be specified for the decision function.

### Disadvantages of SVMs:

*   If the number of features is much greater than the number of samples, avoiding overfitting by choosing appropriate Kernel functions and regularization terms is crucial.
*   SVMs do not directly provide probability estimates; these are calculated using an expensive five-fold cross-validation.

### SVM Classification (SVC)

SVC solves the following primal problem:

$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n \xi_i$
subject to $y_i(w \cdot x_i - b) \ge 1 - \xi_i$ and $\xi_i \ge 0$ for all $i$.

Intuitively, this aims to maximize the margin (by minimizing $w^T w$) while incurring a penalty ($\xi_i$) when a sample is misclassified or falls within the margin boundary. The penalty term $C$ controls the strength of this penalty.

*   [Support Vector Machine Explained](https://www.saedsayad.com/support_vector_machine.htm)

### SVM Regression (SVR)

SVR solves the following primal problem:

$\min_{w, b, \xi, \xi^*} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n (\xi_i + \xi_i^*)$
subject to $y_i - (w \cdot x_i - b) \le \epsilon + \xi_i$
$(w \cdot x_i - b) - y_i \le \epsilon + \xi_i^*$
$\xi_i, \xi_i^* \ge 0$ for all $i$.

Here, samples are penalized if their prediction is at least $\epsilon$ away from their true target. The penalty term $C$ controls the strength of this penalty.

*   [Support Vector Regression Explained](https://www.saedsayad.com/support_vector_machine_reg.htm)

### Unbalanced Problems

In problems where certain classes or individual samples need more importance, parameters `class_weight` and `sample_weight` can be used.

### Kernel Functions

(Further details on various Kernel functions such as linear, polynomial, radial basis function (RBF), sigmoid, etc., would typically be provided here.)

## Nearest Neighbors

The principle behind nearest neighbor methods is to find a predefined number of training samples closest in distance to a new point and predict its label from these neighbors.

*   The number of samples can be a user-defined constant (k-nearest neighbor learning) or vary based on the local density of points (radius-based neighbor learning).
*   The distance can be any metric measure; Euclidean distance is the most common.
*   Neighbors-based methods are known as non-generalizing machine learning methods as they simply "remember" all their training data.

### Advantages

*   Despite its simplicity, nearest neighbors has been successful in many classification and regression problems, including handwritten digits and satellite image scenes.
*   Being a non-parametric method, it is often successful in classification situations where the decision boundary is very irregular.

### Classification

*   **Uniform Weights:** The value assigned to a query point is computed from a simple majority vote of the nearest neighbors (`weights = 'uniform'`).
*   **Distance-based Weights:** It is often better to weight neighbors such that nearer neighbors contribute more to the fit. `weights = 'distance'` assigns weights proportional to the inverse of the distance from the query point.

### Regression

(Further details on nearest neighbors for regression would typically be provided here, likely focusing on averaging the values of the nearest neighbors.)

## Decision Trees

Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression.

*   The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.
*   A tree can be seen as a piecewise constant approximation.

### Advantages of Decision Trees:

*   Simple to understand and interpret. Trees can be visualized.
*   Requires little data preparation (doesn't need normalization).
*   Able to handle both numerical and categorical data. (Note: The scikit-learn implementation does not directly support categorical variables in its current form; refer to algorithms for more information).
*   Possible to validate a model using statistical tests, which helps account for the model's reliability.
*   Performs well even if its assumptions are somewhat violated by the true model from which the data were generated.

### Disadvantages of Decision Trees:

*   Decision-tree learners can create overly complex trees that do not generalize the data well (overfitting).
*   Decision trees can be unstable because small variations in the data might result in a completely different tree being generated.
*   Predictions of decision trees are neither smooth nor continuous, making them not ideal for extrapolation.
*   Learning an optimal decision tree is an NP-complete problem, meaning there's no guarantee of returning the globally optimal decision tree.
*   Decision tree learners create biased trees if some classes dominate. It is recommended to balance the dataset prior to fitting with a decision tree.

### Classification

*   A decision tree trained on the Iris dataset is a common example.
*   Uses metrics like the Gini score for splitting nodes.

### Regression

*   **Complexity Control:** Parameters like `max_depth` are used to control the complexity of the tree to prevent overfitting.

## Neural Network Models

### Multi-layer Perceptron (MLP)

Multi-layer Perceptron (MLP) is a supervised learning algorithm that learns a function $f: R^D \to R^O$ by training on a dataset.

*   It can learn a non-linear function approximator for either classification or regression.
*   Between the input and the output layer, there can be one or more non-linear layers, called hidden layers.

### Neuron Transformation

Each neuron in the hidden layer transforms the values from the previous layer with a weighted linear summation:

$z = W^T x + b$

followed by a non-linear activation function:

$a = g(z)$

### Advantages of MLPs:

*   Capability to learn non-linear models.
*   Capability to learn models in real-time (on-line learning) using `partial_fit`.

### Disadvantages of MLPs:

*   MLP with hidden layers have a non-convex loss function with more than one local minimum. Different random weight initializations can lead to different validation accuracy.
*   MLP requires tuning a number of hyperparameters, such as the number of hidden neurons, layers, and iterations.
*   MLP is sensitive to feature scaling.

### MLP Function Learning

MLP learns the function:

$f(x) = G(W_2 G(W_1 x + b_1) + b_2)$

Where $W_1$, $W_2$ represent the weights of the input layer and hidden layer, respectively; and $b_1$, $b_2$ represent the bias added to the hidden layer and the output layer, respectively.

The default activation function is the hyperbolic tangent (`tanh`).
*   [Hyperbolic Tangent Function](https://www.mathworks.com/help/matlab/ref/tanh.html)

### Output Activation Functions

*   For binary classification, $f(x)$ passes through the logistic function (sigmoid).
*   For more than two classes, $f(x)$ passes through the softmax function.
*   In regression, the output remains as $f(x)$; thus, the output activation function is just the identity function.

*   [Logistic Function](https://en.wikipedia.org/wiki/Logistic_function)
*   [A Guide to Softmax Activation Function](https://www.singlestore.com/blog/a-guide-to-softmax-activation-function/)

### Loss Functions for MLPs

*   **For classification, MLP uses Average Cross-Entropy.** In the binary case, it is given as:

    $L = -\frac{1}{N} \sum_{i=1}^N [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$

    *   [Cross-Entropy in Machine Learning](https://www.shopdev.co/blog/cross-entropy-in-machine-learning)

*   **For regression, MLP uses the Mean Squared Error (MSE) loss function,** written as:

    $L = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$

    *   [Cross-Entropy in Machine Learning](https://www.shopdev.co/blog/cross-entropy-in-machine-learning)

### Optimization for MLPs

To find the model parameters, the regularized training error is minimized:

$E(w, b) = L(w, b) + \alpha R(w)$

MLP trains using optimizers like Stochastic Gradient Descent (SGD) or Adam.

*   **Stochastic Gradient Descent (SGD):** Updates parameters using the gradient of the loss function with respect to a parameter that needs adaptation:

    $\omega \leftarrow \omega - \eta \nabla_\omega (L(\omega, b) + R(\omega))$

    Where $\eta$ is the learning rate, which controls the step-size in the parameter space search. $L$ is the loss function, and $R(\omega)$ is the regularization term.

*   **Adam:** Similar to SGD, it is a stochastic optimizer but can automatically adjust the amount to update parameters based on adaptive estimates of lower-order moments.

Both SGD and Adam support online and mini-batch learning.

## Other Loss Functions

*   **Hinge Loss**
*   **Perceptron Loss**
*   **Log Loss**
*   **Squared Error Loss**
*   **Epsilon-Insensitive Loss**

## Ensemble Methods

Ensemble methods combine multiple learning algorithms to obtain better predictive performance than could be obtained from any of the constituent learning algorithms alone.

*   **Voting**
*   **Stacking**
*   **Boosting**
*   **Bagging**

### Voting Classifier

The idea is to combine conceptually different machine learning classifiers and use a majority vote or the average predicted probabilities (soft vote) to predict the class labels.

*   Such a classifier can be useful for a set of equally well-performing models to balance out their individual weaknesses.
*   **Example:**
    *   Classifier 1 -> Class 1
    *   Classifier 2 -> Class 1
    *   Classifier 3 -> Class 2
    *   A `VotingClassifier` (with `voting='hard'`) would classify the sample as "Class 1" based on the majority class label.

### Stacking

The predictions of each individual estimator are stacked together and used as input to a final estimator to compute the final prediction.

*   This final estimator is trained through cross-validation.
*   [An introduction to stacked generalization](https://wolpert.readthedocs.io/en/latest/user_guide/intro.html)

### Boosting

Boosting is an ensemble meta-algorithm that converts weak learners into strong ones.

1.  Form a large set of simple features.
2.  Initialize weights for training images.
3.  For T rounds:
    1.  Normalize the weights.
    2.  For available features from the set, train a classifier using a single feature and evaluate the training error.
    3.  Choose the classifier with the lowest error.
    4.  Update the weights of the training images: increase if classified wrongly by this classifier, decrease if correctly.
4.  Form the final strong classifier as the linear combination of the T classifiers (coefficient larger if training error is small).

*   [Boosting (machine learning) - Wikipedia](https://en.wikipedia.org/wiki/Boosting_(machine_learning))

### Bagging (Bootstrap Aggregating)

Bagging aims to reduce variance in a model.

*   A random sample of data in a training set is selected with replacement.
*   After several data samples are generated, these weak models are then trained independently.
*   The average or majority of those predictions yield a more accurate estimate.
*   [Bagging - IBM](https://www.ibm.com/)

### Random Forest

In a random forest, each tree in the ensemble is built from a sample drawn with replacement (i.e., a bootstrap sample) from the training set.

*   Furthermore, when splitting each node during the construction of a tree, the best split is found either from all input features or a random subset of size `max_features`.
*   [Random Forest Algorithm](https://www.turing.com/kb/random-forest-algorithm)

## Unsupervised Learning

### Clustering

Clustering is a data mining technique that groups unlabeled data based on their similarities or differences.

### Dimensionality Reduction

Dimensionality reduction reduces the number of data inputs to a manageable size while preserving the integrity of the dataset as much as possible.

*   [Dimensionality Reduction - IBM](https://www.ibm.com/)

### Comparison of Clustering Algorithms in Scikit-learn

(A comparison chart or detailed discussion of various scikit-learn clustering algorithms would typically be included here.)

### K-means Clustering

The k-means algorithm divides a set of N samples $X$ into K disjoint clusters $C$, each described by the mean $\mu_i$ of the samples in the cluster. These means are commonly called the cluster "centroids."

*   The K-means algorithm aims to choose centroids that minimize the inertia, or within-cluster sum-of-squares criterion:

    $\sum_{i=0}^n \min_{\mu_j \in C} (\|x_i - \mu_j\|^2)$

#### Mini-Batch K-Means

Mini-batches are subsets of the input data, randomly sampled in each training iteration, making it more efficient for large datasets.

### Hierarchical Clustering

Hierarchical clustering is a general family of clustering algorithms that build nested clusters by merging or splitting them successively.

*   This hierarchy of clusters is represented as a tree (or dendrogram).
*   The root of the tree is the unique cluster that gathers all samples, with leaves being clusters containing only one sample.

#### Agglomerative Hierarchical Clustering

Uses a bottom-up approach: each observation starts in its own cluster, and clusters are successively merged.

The **linkage criteria** determine the metric used for the merge strategy:

*   **Ward:** Minimizes the sum of squared differences within all clusters; a variance-minimizing approach.
*   **Maximum (Complete) Linkage:** Minimizes the maximum distance between observations of pairs of clusters.
*   **Average Linkage:** Minimizes the average of the distances between all observations of pairs of clusters.
*   **Single Linkage:** Minimizes the distance between the closest observations of pairs of clusters.

#### Characteristics of Agglomerative Clustering

*   Agglomerative clustering can lead to uneven cluster sizes ("rich get richer" behavior).
*   Single linkage is often the worst strategy.
*   Ward linkage typically gives the most regular sizes.
*   For non-Euclidean metrics, average linkage is a good alternative.
*   Single linkage can be computed very efficiently, making it useful for larger datasets.

### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

DBSCAN defines clusters based on the density of points.

*   A **core sample** is a sample in the dataset such that there exist `min_samples` other samples within a distance of `eps`, defined as neighbors of the core sample. This indicates the core sample is in a dense area.
*   A **cluster** is a set of core samples built by recursively taking a core sample, finding all its core sample neighbors, and so on.
*   **Outliers** are samples that are neither core samples nor reachable from core samples.

### Clustering Performance Evaluation

#### When the Ground Truth is Known:

*   Rand Index
*   Homogeneity, Completeness, and V-measure
*   Fowlkes-Mallows Scores

#### When the Ground Truth is Unknown:

*   Silhouette Coefficient
*   Davies-Bouldin Index

### Clustering Performance Evaluation - Rand Index

Given ground truth class assignments `labels_true` and clustering algorithm assignments `labels_pred`.

*   **Rand Index ($RI$):** Measures the similarity of the two assignments, ignoring permutations.

    $RI = \frac{a+b}{C_2^n}$

    Where:
    *   $a$: Number of pairs of elements that are in the same set in `labels_true` and in the same set in `labels_pred`.
    *   $b$: Number of pairs of elements that are in different sets in `labels_true` and in different sets in `labels_pred`.
    *   $C_2^n$: Total number of possible pairs in the dataset.

*   **Advantages:**
    1.  Interpretability.
    2.  Random (uniform) label assignments have an adjusted Rand index score close to 0.
    3.  Bounded range in [0, 1] for unadjusted RI & [-1, 1] for adjusted RI.
    4.  No assumption is made on the cluster structure.
*   **Disadvantages:**
    1.  Requires knowledge of the ground truth classes.

### Clustering Performance Evaluation - Homogeneity, Completeness and V-measure

*   **Homogeneity:** Each cluster contains only members of a single class.
*   **Completeness:** All members of a given class are assigned to the same cluster.
*   **V-measure:** The harmonic mean of homogeneity and completeness, with $\beta$ defaulting to 1.0.

*   **Advantages:**
    1.  Bounded scores: 0.0 is as bad as it can be, 1.0 is a perfect score.
    2.  Qualitatively analyzed in terms of homogeneity and completeness.
    3.  No assumption is made on the cluster structure.
*   **Disadvantages:**
    1.  Not normalized with regards to random labeling.
    2.  Random labeling won't yield zero scores, especially when the number of clusters is large. (This problem can usually be ignored when the number of samples is more than a thousand and the number of clusters is less than 10.)

### Clustering Performance Evaluation - Fowlkes-Mallows Scores

The Fowlkes-Mallows index (`sklearn.metrics.fowlkes_mallows_score`) can be used when the ground truth class assignments of the samples are known.

*   The Fowlkes-Mallows score (FMI) is defined as the geometric mean of the pairwise precision and recall.

*   **Advantages:**
    1.  Random (uniform) label assignments have an FMI score close to 0.
    2.  Upper-bounded at 1.
    3.  No assumption is made on the cluster structure.
*   **Disadvantages:**
    1.  Requires knowledge of the ground truth classes.

### Clustering Performance Evaluation - Silhouette Coefficient

A higher Silhouette Coefficient score relates to a model with better-defined clusters.

*   The Silhouette Coefficient is defined for each sample and is composed of two scores:
    *   $a$: The mean distance between a sample and all other points in the same class.
    *   $b$: The mean distance between a sample and all other points in the next nearest cluster.
*   The Silhouette Coefficient $s$ for a single sample is then given as:

    $s = \frac{b - a}{\max(a, b)}$

*   **Advantages:**
    1.  The score is bounded between -1 for incorrect clustering and +1 for highly dense clustering. Scores around zero indicate overlapping clusters.
    2.  The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.
*   **Disadvantages:**
    1.  The Silhouette Coefficient is generally higher for convex clusters than other concepts of clusters, such as density-based clusters like those obtained through DBSCAN.

### Clustering Performance Evaluation - Davies-Bouldin Index

A lower Davies-Bouldin index relates to a model with better separation between the clusters.

*   This index signifies the average "similarity" between clusters, where similarity compares the distance between clusters with the size of the clusters themselves.
*   Zero is the lowest possible score. Values closer to zero indicate a better partition.

*   **Advantages:**
    1.  The computation of Davies-Bouldin is simpler than that of Silhouette scores.
    2.  The index is solely based on quantities and features inherent to the dataset, as its computation only uses point-wise distances.
*   **Disadvantages:**
    1.  The Davies-Bouldin index is generally higher for convex clusters than other concepts of clusters, such as density-based clusters like those obtained from DBSCAN.
    2.  The usage of centroid distance limits the distance metric to Euclidean space.

## Dimensionality Reduction

Dimensionality reduction is a technique used to reduce the number of features in a dataset while retaining as much of the important information as possible.

*   It transforms high-dimensional data into a lower-dimensional space that still preserves the essence of the original data.
*   [Dimensionality Reduction - GeeksforGeeks](https://www.geeksforgeeks.org/dimensionality-reduction/)

### Principal Component Analysis (PCA)

PCA is used to decompose a multivariate dataset into a set of successive orthogonal components that explain a maximum amount of the variance.

*   Example: IRIS dataset visualization.

### Linear Discriminant Analysis (LDA) for Dimensionality Reduction

LDA, in contrast to PCA, is a supervised method, using known class labels.

*   Linear Discriminant Analysis (LDA) tries to identify attributes that account for the most variance *between* classes.
*   [Implementing a Linear Discriminant Analysis (LDA) from scratch in Python](https://sebastianraschka.com/Articles/2014_python_lda.html)

### Manifold Learning

Manifold learning is an approach to non-linear dimensionality reduction.

*   Manifold Learning can be thought of as an attempt to generalize linear frameworks like PCA to be sensitive to non-linear structure in data.

### Manifold Learning - Isomap

One of the earliest approaches to manifold learning.

*   Isomap can be viewed as an extension of Multi-dimensional Scaling (MDS) or Kernel PCA.
*   Isomap seeks a lower-dimensional embedding that maintains geodesic distances between all points.
*   **Geodesic distance:** Between two vertices, it's the length (in terms of the number of edges) of the shortest path between the vertices.

### Manifold Learning - t-distributed Stochastic Neighbor Embedding (t-SNE)

The basic idea is to convert similarities between data points in the high-dimensional space into probabilities, and then map these probabilities to a lower-dimensional space in a way that preserves the relationships between the data points as much as possible.

*   This allows t-SNE to be particularly sensitive to local structure.
*   [Easy explanation of Dimensionality Reduction and Techniques](https://aurigait.com/blog/blog-easy-explanation-of-dimensionality-reduction-and-techniques/)

#### Procedure for t-SNE:

1.  **Compute pairwise similarities:** Pairwise similarities between data points in the high-dimensional space are computed (often using a Gaussian distribution).
2.  **Construct probability distributions:** From these pairwise similarities, probability distributions are constructed for each data point.
3.  **Initialize embedding:** An initial embedding of the data points in the low-dimensional space is randomly generated. Each data point is assigned a position.
4.  **Compute similarity in low-dimensional space:** Pairwise similarities between data points are computed in the low-dimensional space (often using a Student’s t-distribution).
5.  **Optimize embedding:** The goal is to minimize the divergence between the pairwise similarities in the high-dimensional space and the low-dimensional space (typically using gradient descent).
6.  **Convergence:** The optimization process continues until a stopping criterion is met.
7.  **Visualization:** The final low-dimensional embedding can be visualized.

*   [Easy explanation of Dimensionality Reduction and Techniques](https://aurigait.com/blog/blog-easy-explanation-of-dimensionality-reduction-and-techniques/)

#### Advantages of t-SNE:

1.  Revealing the structure at many scales on a single map.
2.  Revealing data that lie in multiple, different manifolds or clusters.
3.  Reducing the tendency to crowd points together at the center.

#### Disadvantages of t-SNE:

1.  t-SNE is computationally expensive; it can take several hours on million-sample datasets where PCA finishes in seconds or minutes.
2.  The Barnes-Hut t-SNE method is limited to two or three-dimensional embeddings.
3.  The algorithm is stochastic, and multiple restarts with different seeds can yield different embeddings. However, it is legitimate to pick the embedding with the least error.
4.  Global structure is not explicitly preserved. This problem is mitigated by initializing points with PCA (using `init='pca'`).

*   [Easy explanation of Dimensionality Reduction and Techniques](https://aurigait.com/blog/blog-easy-explanation-of-dimensionality-reduction-and-techniques/)

### Manifold Learning - Uniform Manifold Approximation and Projection (UMAP)

UMAP constructs a low-dimensional representation of the data that preserves both the global and local structure of the high-dimensional space.

*   UMAP uses a graph-based approach to build a topological representation of the data, which is then embedded in a low-dimensional space using stochastic gradient descent.
*   [Easy explanation of Dimensionality Reduction and Techniques](https://aurigait.com/blog/blog-easy-explanation-of-dimensionality-reduction-and-techniques/)

#### Procedure for UMAP:

1.  **Compute the pairwise distances:** Calculate the pairwise distances between data points in the high-dimensional space.
2.  **Construct a fuzzy simplicial set:** Based on pairwise distances, a fuzzy simplicial set is constructed, representing the local neighborhood structure.
3.  **Optimize the low-dimensional embedding:** A random low-dimensional embedding is initially generated. UMAP then uses stochastic gradient descent to minimize the discrepancy between high-dimensional and low-dimensional pairwise similarities.
4.  **Construct a graph and perform graph layout:** UMAP constructs a graph representation from the low-dimensional embedding, capturing global data structure.
5.  **Refine the embedding:** A refinement step adjusts data point positions based on graph structure and data connectivity.
6.  **Convergence:** Optimization and refinement steps iterate until convergence.
7.  **Visualization and analysis:** The low-dimensional representation aims to preserve neighborhood relationships and global structure.

*   [Easy explanation of Dimensionality Reduction and Techniques](https://aurigait.com/blog/blog-easy-explanation-of-dimensionality-reduction-and-techniques/)