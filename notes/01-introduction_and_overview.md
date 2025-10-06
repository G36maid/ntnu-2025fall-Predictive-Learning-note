# 01-Introduction and Overview

This note summarizes the key concepts from Lecture Set 1 of Predictive Learning from Data.

**Source:** Cherkassky, Vladimir, and Filip M. Mulier. *Learning from data: concepts, theory, and methods*. John Wiley & Sons, 2007. (Revised by Dr. Hsiang-Han Chen)

*(Please do not distribute without author's permission.)*

## 1.1 Overview: What is this course about?

### Uncertainty and Learning

This section introduces the fundamental concepts of uncertainty and learning, exploring their relevance in decision-making and scientific inquiry.

*   **Decision making under uncertainty:** How do we make choices when outcomes are not guaranteed?
*   **Biological learning (adaptation):** Observing how living systems learn and adapt.
*   **Plausible (uncertain) inference:** Drawing conclusions that are likely but not certain.
*   **Induction in Statistics and Philosophy:**
    *   **Example 1: Many old men are bald.**
    *   **Example 2: Sun rises on the East every day.**

#### Exploring "Many old men are bald"

*   **Psychological Induction:** An inductive statement based on experience, with a predictive aspect but no scientific explanation.
*   **Statistical View:** Treating "lack of hair" as a random variable and estimating its distribution from past observations (training sample).
*   **Philosophy of Science Approach:** Seeking a scientific theory to explain the phenomenon. A true theory needs to make non-trivial predictions, not just explanations.

### Conceptual Issues

Any theory or model has two key aspects:

1.  **Explanation of past data (observations):** How well does the model account for what has already happened?
2.  **Prediction of future (unobserved) data:** How well does the model forecast what will happen?

A model capable of achieving both goals perfectly is not possible. Important issues to address include:

*   The quality of explanation and prediction.
*   Is good prediction possible at all?
*   If two models explain past data equally well, which one is better?
*   How to measure model complexity?

### Beliefs vs. Scientific Theories

The "demarcation problem" in philosophy addresses how to distinguish between science and non-science. For example, considering the statement "Men have lower life expectancy than women," various explanations might be offered:

*   Because they choose to do so.
*   Because they make more money (on average) and experience higher stress managing it.
*   Because they engage in risky activities.
*   ...and so on.

### Philosophical Connections: Induction

Induction, as defined by the Oxford English dictionary, is "the process of inferring a general law or principle from the observations of particular instances." This is clearly related to Predictive Learning, as all science and most human knowledge involves some form of induction.

The core question is: How to form 'good' inductive theories? This leads to the concept of **inductive principles** or general rules.

#### Occam's Razor

William of Ockham's principle, "entities should not be multiplied beyond necessity," suggests reducing assumptions to their minimum. This is a fundamental inductive principle.

*   [Occam's razor - Wikipedia](https://en.wikipedia.org/wiki/Occam%27s_razor)
*   [Occam's Razor Definition & Meaning - Market Business News](https://marketbusinessnews.com/financial-glossary/ockhams-razor-definition-meaning/#:~:text=Regarding%20what%20became%20known%20as,that%20we%20form%20and%20administer.)

## 1.2 Prerequisites and Expected Outcomes

### Learning = Generalization

This course focuses on learning as generalization, covering its concepts and related issues.

### Scientific / Technical Outcomes

*   **Math theory:** Statistical Learning Theory (aka VC-theory).
*   **Conceptual basis:** Understanding the underlying principles for various learning algorithms.

### Methodological Outcomes

*   How to use available statistical/machine learning/data mining software.
*   How to compare prediction accuracy of different learning algorithms.
*   Developing an understanding of whether modeling results are due to skill or chance.

### Practical Applications

*   Financial engineering
*   Biomedical + Life Sciences
*   Security
*   Image recognition, etc.

### Modeling Financial Data on Yahoo! Finance

*   **Real Data:** Analyzing daily price changes of SP500, calculated as $X(t) = (Z(t) - Z(t-1)) / Z(t-1) * 100\%$, where $Z(t)$ is the closing price.
*   **Question:** Is the stock market truly random?
*   **Modeling Assumption:** Price changes $X$ are independent and identically distributed (i.i.d.), which leads to analytic relationships verifiable with empirical data.

#### Understanding Daily Price Changes

*   **Histogram:** An estimated Probability Density Function (PDF) derived from data.
    *   Example: Histograms of 5 and 30 bins to model a Normal distribution N(0,1).
    *   Also includes mean and standard deviation estimated from data.
*   **Note:** Histograms represent empirical PDF, with the y-axis typically scaled in frequency (%).

## 1.3 Big Data and Scientific Discovery

### Historical Example: Ulisse Aldrovandi (16th Century)

*   Authored "Natural History of Snakes."

### Promise of Big Data

*   Often presented as "technical fairy tales" driven by marketing.
*   **Core promise:** `software program + DATA → knowledge`.
*   **Implies:** More Data → More Knowledge.
*   The idea: "Yes-we-Can!"

### Examples from Life Sciences… (Cautionary Tale)

*   Duke biologists "discovered" an unusual link between a popular singer (Lady Gaga) and a new species of fern based on a sequence GAGA found in the fern’s DNA base pairs. This highlights the potential for spurious correlations in large datasets.

### Scientific Discovery

Scientific discovery combines ideas/models with facts/data.

*   **First-principle knowledge:** `hypothesis → experiment → theory`
    *   Deterministic, causal, intelligible models.
*   **Modern data-driven discovery:** `software program + DATA → knowledge`
    *   Statistical, complex systems.
*   Many methodological differences exist between these two approaches.

### Invariants of Scientific Knowledge

Essential characteristics of scientific knowledge:

*   **Intelligent questions**
*   **Non-trivial predictions**
*   **Clear limitations/constraints**

All these require human intelligence, which might be missing or lost in the era of Big Data.

### Historical Example: Planetary Motions

*   **Ptolemaic system (geocentric) vs. Copernican system (heliocentric).**
*   **Tycho Brahe (16th century):** Measured positions of planets to support views with experimental data.
*   **Johannes Kepler:** Used Tycho's extensive data to discover three remarkably simple laws of planetary motion.

#### First Kepler's Law

*   The Sun lies in the plane of orbit. Positions can be represented as (x,y) pairs.
*   An orbit is an ellipse, with the Sun at one focus.

#### Second Kepler's Law

*   The radius vector from the Sun to the planet sweeps out equal areas in the same time intervals.

#### Third Kepler's Law

*   For any planet, $P^2 \sim D^3$, where $P$ is the orbit period and $D$ is the orbit size (half-diameter).

| Planet    | P (years) | D (AU) | P²      | D³      |
| :-------- | :-------- | :----- | :------ | :------ |
| Mercury   | 0.24      | 0.39   | 0.058   | 0.059   |
| Venus     | 0.62      | 0.72   | 0.38    | 0.39    |
| Earth     | 1.00      | 1.00   | 1.00    | 1.00    |
| Mars      | 1.88      | 1.53   | 3.53    | 3.58    |
| Jupiter   | 11.90     | 5.31   | 142.0   | 141.00  |
| Saturn    | 29.30     | 9.55   | 870.0   | 871.00  |

### Empirical Scientific Theory

*   Kepler’s Laws could:
    *   Explain experimental data.
    *   Predict new data (e.g., other planets).
    *   **BUT did not explain *why* planets move.**
*   **Popular explanation:** Planets move because invisible angels beat wings behind them (a belief).
*   **First-principle scientific explanation:** Galileo and Newton discovered laws of motion and gravity that provided the underlying physical explanation for Kepler’s laws.

## 1.4 Related Data Modeling Methodologies

### Scientific Knowledge

*   **Knowledge:** Stable relationships between facts and ideas (mental constructs).
*   **Classical first-principle knowledge:** Rich in ideas, relatively few facts (amount of data), simple relationships.

### Growth of Empirical Knowledge

*   Huge growth of data in the 20th century (computers and sensors).
*   Focus on complex systems (engineering, life sciences, and social).
*   Classical first-principles science is often inadequate for empirical knowledge.
*   **Need for new Methodology:** How to estimate good predictive models from noisy data?

### Different Types of Knowledge

*   **Scientific (first-principles, deterministic)**
*   **Empirical (uncertain, statistical)**
*   **Metaphysical (beliefs)**

The boundaries between these types of knowledge are often poorly understood.

### Handling Uncertainty and Risk (Part 1)

*   **Ancient times:** Uncertainty was often attributed to divine will or fate.
*   **Probability for quantifying uncertainty:**
    *   **Degree-of-belief:** Subjective probability.
    *   **Frequentist:** Based on observed frequencies (Cardano-1525, Pascale, Fermat).
*   **Newton and causal determinism:** Universe as a predictable machine.
*   **Probability theory and statistics (20th century):** Formalized the study of randomness.
*   **Modern classical science (A. Einstein):** Goal of science is estimating a true model or system identification.

### Handling Uncertainty and Risk (Part 2)

*   **Making decisions under uncertainty:** Involves risk management, adaptation, and intelligence.
*   **Probabilistic approach:**
    *   Estimate probabilities of future events.
    *   Assign costs and minimize expected risk.
*   **Risk minimization approach:**
    *   Apply decisions to known past events.
    *   Select the one minimizing expected risk.

### Summary: Knowledge & Modeling Goals

*   **First-principles knowledge:** Deterministic relationships between a few concepts (variables).
*   **Importance of empirical knowledge:** Statistical in nature, usually many input variables.
*   **Goal of modeling:** To act/perform well, rather than solely for system identification.

### Other Related Methodologies

Estimation of empirical dependencies is commonly addressed in many fields:

*   Statistics
*   Data Mining
*   Machine Learning
*   Neural Networks
*   Signal Processing, etc.

Each field often has its own methodological bias and terminology, leading to confusion.

*   **Quotations from popular textbooks:**
    *   **Pattern Recognition:** "concerned with the automatic discovery of regularities in data."
    *   **Data Mining:** "the process of automatically discovering useful information in large data repositories."
    *   **Statistical Learning:** "about learning from data."

All these fields are essentially concerned with estimating predictive models from data.

#### Generic Problem

Estimate (learn) useful models from available data.

Methodologies often differ in terms of:

*   What is considered "useful."
*   Assumptions about available data.
*   Goals of learning.

These important notions are often not well-defined.

### Common Goals of Modeling

*   **Prediction (Generalization)**
*   **Interpretation** (descriptive model)
*   **Human decision-making** using both prediction and interpretation.
*   **Information retrieval:** Predictive or descriptive modeling of an unspecified subset of available data.

**Note:** These goals are usually ill-defined. Formalization of these goals in the context of application requirements is **THE MOST IMPORTANT** aspect of 'data mining.'

### Three Distinct Methodologies

1.  **Statistical Estimation:** From classical statistics and function approximation.
2.  **Predictive Learning (~ machine learning):**
    *   Practitioners in machine learning / neural networks.
    *   Vapnik-Chervonenkis (VC) theory for estimating predictive models from empirical (finite) data samples.
3.  **Data Mining:** Exploratory data analysis, e.g., selecting a subset of available (large) dataset with interesting properties.

## 1.5 General Experimental Procedure for Estimating Models from Data

This procedure is iterative and complex at each step, with the estimated model depending on all previous steps. It's crucial to acknowledge that often, we deal with observational data rather than data from designed experiments.

1.  **Statement of the Problem:** Clearly define what needs to be solved.
2.  **Hypothesis Formulation (Problem Formalization):** This step differs from classical statistics in its approach.
3.  **Data Generation/Experiment Design:** How the data is collected or generated.
4.  **Data Collection and Preprocessing:**
    *   Preprocessing is essential for observational data.
    *   **Basic preprocessing includes:**
        *   Summary univariate statistics (mean, standard deviation, min, max, range, boxplot) performed independently for each input/output.
        *   Detection (removal) of outliers.
        *   Scaling of input/output variables (may be necessary for some learning algorithms).
    *   Visual inspection of data, while tedious, is often very useful.
5.  **Model Estimation (learning):** The core process of training a model.
6.  **Model Interpretation, Model Assessment, and Drawing Conclusions:** Understanding the model, evaluating its performance, and deriving insights.

### Cultural + Ethical Aspects

Cultural and business aspects can significantly affect:

*   Problem formalization.
*   Data access/sharing (e.g., in life sciences).
*   Model interpretation.

A possible (idealistic) solution involves adopting a common methodology, which is critical for interdisciplinary projects.

### Honest Disclosure of Results (Publication Bias)

*   **Modern drug studies example:** A review of 74 studies submitted to the FDA found 38 were positive. All but one of the positive studies were published. Most studies with negative or questionable results were *not* published. (Source: The New England Journal of Medicine, WSJ Jan 17, 2008).
*   **Publication bias:** A common issue in modern research, where positive results are more likely to be published than negative or inconclusive ones.