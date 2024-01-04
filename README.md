# Fairness-Aware Neural Renyi Minimization for Continuous Features

This work has been accepted at the IJCAI 2020 conference.

https://www.ijcai.org/proceedings/2020/0313.pdf

The past few years have seen a dramatic rise of academic and societal interest in fair machine learning.
While plenty of fair algorithms have been proposed
recently to tackle this challenge for discrete variables, only a few ideas exist for continuous ones.
The objective in this paper is to ensure some independence level between the outputs of regression models and any given continuous sensitive
variables. For this purpose, we use the HirschfeldGebelein-Renyi (HGR) maximal correlation coef- ´
ficient as a fairness metric. We propose to minimize the HGR coefficient directly with an adversarial neural network architecture. The idea is to
predict the output Y while minimizing the ability
of an adversarial neural network to find the estimated transformations which are required to predict the HGR coefficient. We empirically assess and
compare our approach and demonstrate significant
improvements on previously presented work in the
field.


<p align="center">
  <img src="https://github.com/fairml-research/HGR_NN/blob/main/img.png?raw=true" width="550" title="hover text">
</p>

