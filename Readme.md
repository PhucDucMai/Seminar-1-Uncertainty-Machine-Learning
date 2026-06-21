# Deep Ensembles for Model Uncertainty Estimation in Deep Learning: A Literature Review

## Seminar Information

* **Course:** Seminar 1
* **Research Area:** Machine Learning and Deep Learning
* **Lecturer: Professor. Nguyen Hung Son

---

## Overview

Deep Neural Networks (DNNs) have achieved state-of-the-art performance across numerous applications. However, they often produce overconfident predictions, even when encountering unfamiliar or out-of-distribution data.

This limitation poses significant challenges for safety-critical domains such as autonomous driving, medical diagnosis, industrial inspection, and disaster response.

Uncertainty Quantification (UQ) aims to estimate the reliability of model predictions. Among different sources of uncertainty, this seminar focuses specifically on **Model Uncertainty (Epistemic Uncertainty)**.

Model uncertainty arises from limited knowledge of the model and insufficient training data and can potentially be reduced by collecting additional information.

---

## Research Question

> Which model uncertainty estimation method provides more reliable predictive uncertainty in deep learning models?

---

## Research Objectives

The objectives of this seminar are:

* Review the theoretical foundations of model uncertainty in deep learning.
* Investigate representative uncertainty estimation methods.
* Compare Monte Carlo Dropout and Deep Ensembles.
* Analyze their effectiveness in calibration and out-of-distribution detection.
* Identify current challenges and future research directions.

---

## Research Scope

This seminar focuses exclusively on **Model Uncertainty (Epistemic Uncertainty)**.

The following methods will be investigated:

* Monte Carlo Dropout (MC Dropout)
* Deep Ensembles

The comparison will be conducted from the following perspectives:

* Theoretical foundations
* Computational cost
* Scalability
* Calibration performance
* Out-of-Distribution (OOD) detection capability
* Practical applicability

**Out of Scope:**

* Data Uncertainty (Aleatoric Uncertainty)
* Bayesian Neural Networks with complex inference methods
* Large Language Models (LLMs)
* Multi-modal uncertainty estimation

---

## Literature Review Methodology

The literature review will follow a snowballing approach:

1. Identify foundational papers.
2. Perform backward citation analysis.
3. Perform forward citation analysis.
4. Review recent studies published between 2020 and 2025.

---

## Foundational Papers

* Kendall, A., & Gal, Y. (2017). *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?*
* Wilson, A. G., & Izmailov, P. (2020). *Bayesian Deep Learning and a Probabilistic Perspective of Generalization.*
* Gawlikowski, J., et al. (2023). *A Survey of Uncertainty in Deep Neural Networks.*
* He, W., et al. (2023). *A Survey on Uncertainty Quantification Methods for Deep Learning.

---

## Monte Carlo Dropout

* Gal, Y., & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.*
* Ovadia, Y., et al. (2019). *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty under Dataset Shift.*

---

## Deep Ensembles

* Lakshminarayanan, B., et al. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.*
* Wenzel, F., et al. (2020). *How Good is the Bayes Posterior in Deep Neural Networks Really?*

---

## Recent Papers (2022–Present)

* Weiss, M., & Tonella, P. (2023). *Uncertainty Quantification for Deep Neural Networks: An Empirical Comparison and Usage Guidelines.*
* Fisch, S., et al. (2023). *Sources of Uncertainty in Supervised Machine Learning: A Statistician's View.*