# Deep Ensembles for Model Uncertainty Estimation in Deep Learning: A Literature Review

## Seminar Information

* **Course:** Seminar 1
* **Research Area:** Uncertainty Quantification Machine Learning
* **Supervisor:** Pro. Nguyen Hung Son

## Overview

Deep Neural Networks often produce overconfident predictions, especially when facing distribution shifts or out-of-distribution data. This issue limits their deployment in safety-critical applications.  

While Deep Ensembles is widely considered the gold standard for robust predictive uncertainty estimation, its massive computational and memory costs prohibit real-world deployment on resource-constrained devices. This seminar focuses on Model Uncertainty (Epistemic Uncertainty) and investigates Packed-Ensembles—a parameter-efficient architectural solution that uses grouped convolutions to approximate the performance and diversity of Deep Ensembles at a fraction of the cost.  

The primary goal is to evaluate the trade-off between hardware efficiency and uncertainty quantification reliability under dataset shift.  

---

## Research Question

> How does the Packed-Ensembles architecture trade off parameter efficiency and reliable epistemic uncertainty estimation under dataset shift compared to traditional Deep Ensembles?

---

## Research Objectives

* Contextualize: Understand the theoretical foundations of model uncertainty (distinguishing Epistemic vs. Aleatoric)
* Analyze: Study the traditional Deep Ensembles framework and its computational bottlenecks
* Implement: Investigate the Packed-Ensembles architecture and its use of grouped convolutions for sub-network diversity
* Experiment: Evaluate the effectiveness of Packed-Ensembles under dataset shift (e.g., CIFAR-10-C).
* Critique: Identify architectural limitations (e.g., channel bottlenecks) and propose future research directions

---

## Research Scope

### In Scope

* Model Uncertainty (Epistemic Uncertainty)
* Deep Ensembles & Parameter-Efficient Ensembles (Packed-Ensembles)
* Calibration & Out-of-Distribution Detection
* Dataset Shift Robustness

### Out of Scope

* Data Uncertainty (Aleatoric Uncertainty)
* Bayesian Neural Networks
* Monte Carlo Dropout
* Variational Inference
* Conformal Prediction
* Large Language Models

---

## Research Methodology

The project is executed within a 1-month timeframe, combining a snowball literature review with an empirical study:

1. Literature Review: Identify foundational uncertainty papers and perform backward/forward citation analysis centered around efficient ensemble methods
2. Empirical Experimentation: Utilize the torch-uncertainty open-source library within the PyTorch ecosystem to efficiently train models, varying sub-network counts ($M$) and expansion factors ($\alpha$).
3. Synthesis & Criticism: Summarize findings, analyze the trade-offs, and formulate a 4-6 page double-column short paper.

---

## Core & Foundational Papers
* Main Focus Paper: Laurent, O., et al. (2023). Packed-Ensembles for Efficient Uncertainty Estimation. (NeurIPS 2023).

---

## Foundational Theory:
* Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.
* Kendall, A., & Gal, Y. (2017). What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?

## Related work:
* Arsenii Ashukha, Alexander Lyzhov, Dmitry Molchanov , Dmitry Vetrov (2021). Pitfalls of in-domain uncertainty estimation and ensemble in deep learning
* Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, Dawn Song, Jacob Steinhardt, Justin Gilmer. The Many Faces of Robustness: A Critical Analysis of Out-of-Distribution Generalization (2021)
* Hao Chen, Abhinav Shrivastava (2020). Group Ensemble: Learning an Ensemble of ConvNets in a single ConvNet
* WENCHONG HE, ZHE JIANG, TINGSONG XIAO, ZELIN XU, YUKUN LI (2025). A Survey on Uncertainty Quantification Methods for Deep Learning

## Expected Outcomes

* A short academic paper (4-6 pages, double column) detailing the research findings
* Empirical charts demonstrating how Epistemic Uncertainty behaves when Packed-Ensembles face increasing levels of data corruption
* A critical analysis highlighting the limitations of grouped convolutions in uncertainty estimation, alongside actionable peer reviews for other students' work