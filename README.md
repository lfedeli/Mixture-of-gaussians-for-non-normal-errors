[//]: # (Image References)
[image1]: ./grafo.png "Output"

# Introduction

This project grounds on [1]. The issue is the following: find the best linear regression model that describes  a datasets in which the residuals are non-gaussian distributed. The core calculations use the routines offered by TensorFlow.


# Model generation

The i-th residual is thought to be distributed as the sum of k gaussians:

<a href="https://www.codecogs.com/eqnedit.php?latex=p(\varepsilon^{(i)}&space;|&space;x^{(i)};&space;\nu_h,&space;\tau_h^2&space;)&space;\sim&space;\sum_{h&space;=&space;1}^k&space;\pi_h&space;N(\nu_h,&space;\tau_h^2)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(\varepsilon^{(i)}&space;|&space;x^{(i)};&space;\nu_h,&space;\tau_h^2&space;)&space;\sim&space;\sum_{h&space;=&space;1}^k&space;\pi_h&space;N(\nu_h,&space;\tau_h^2)" title="p(\varepsilon^{(i)} | x^{(i)}; \nu_h, \tau_h^2 ) \sim \sum_{h = 1}^k \pi_h N(\nu_h, \tau_h^2)" /></a>

where <a href="https://www.codecogs.com/eqnedit.php?latex=\pi_{h}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\pi_{h}" title="\pi_{h}" /></a> are weights summing to 1 and we have the following constraint

<a href="https://www.codecogs.com/eqnedit.php?latex=\sum_{h&space;=&space;1}^k&space;\pi_h&space;\nu_h&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sum_{h&space;=&space;1}^k&space;\pi_h&space;\nu_h&space;=&space;0" title="\sum_{h = 1}^k \pi_h \nu_h = 0" /></a>

that guarantees the total mean to be 1. 

Let <a href="https://www.codecogs.com/eqnedit.php?latex=z_{ih}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?z_{ih}" title="z_{ih}" /></a>  be a binary variable equal to 1 when the i-th observation has been generated from th h-component
and to 0 otherwise. The <a href="https://www.codecogs.com/eqnedit.php?latex=z_{ih}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?z_{ih}" title="z_{ih}" /></a>'s are unknown, but if they were known, the so-called complete log-likelihood
would be, up to a constant factor,


<a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{L}(a,b,\nu_h,&space;\tau_h^2,&space;\pi_h)&space;=&space;\sum_{k&space;=&space;1}^h&space;z_{\cdot&space;h}&space;\log&space;\pi_h&space;-\frac{1}{2}&space;\sum_{h&space;=&space;1}^k&space;z_{\cdot&space;h}&space;\log&space;\tau_h^2&space;-\frac{1}{2}&space;\sum_{i&space;=&space;1}^n&space;\sum_{h&space;=&space;1}^k&space;z_{\cdot&space;h}&space;\frac{(y_i&space;-&space;\mu_{ih})^2}{\tau_h^2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathcal{L}(a,b,\nu_h,&space;\tau_h^2,&space;\pi_h)&space;=&space;\sum_{k&space;=&space;1}^h&space;z_{\cdot&space;h}&space;\log&space;\pi_h&space;-\frac{1}{2}&space;\sum_{h&space;=&space;1}^k&space;z_{\cdot&space;h}&space;\log&space;\tau_h^2&space;-\frac{1}{2}&space;\sum_{i&space;=&space;1}^n&space;\sum_{h&space;=&space;1}^k&space;z_{\cdot&space;h}&space;\frac{(y_i&space;-&space;\mu_{ih})^2}{\tau_h^2}" title="\mathcal{L}(a,b,\nu_h, \tau_h^2, \pi_h) = \sum_{k = 1}^h z_{\cdot h} \log \pi_h -\frac{1}{2} \sum_{h = 1}^k z_{\cdot h} \log \tau_h^2 -\frac{1}{2} \sum_{i = 1}^n \sum_{h = 1}^k z_{\cdot h} \frac{(y_i - \mu_{ih})^2}{\tau_h^2}" /></a>

where 

<a href="https://www.codecogs.com/eqnedit.php?latex=z_{\cdot&space;h}&space;=&space;\sum_{i&space;=&space;1}^n&space;z_{ih}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?z_{\cdot&space;h}&space;=&space;\sum_{i&space;=&space;1}^n&space;z_{ih}" title="z_{\cdot h} = \sum_{i = 1}^n z_{ih}" /></a>.

If we replace <a href="https://www.codecogs.com/eqnedit.php?latex=z_{ih}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?z_{ih}" title="z_{ih}" /></a> with <a href="https://www.codecogs.com/eqnedit.php?latex=p_{ih}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_{ih}" title="p_{ih}" /></a> where

<a href="https://www.codecogs.com/eqnedit.php?latex=p_{ih}&space;=&space;\frac{\pi_h&space;\phi(y_i;&space;\mu_{ih}.\tau_h^2)}{\sum_{g=1}^k&space;\pi_g&space;\phi(y_i;&space;\mu_{gh}.\tau_g^2)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_{ih}&space;=&space;\frac{\pi_h&space;\phi(y_i;&space;\mu_{ih}.\tau_h^2)}{\sum_{g=1}^k&space;\pi_g&space;\phi(y_i;&space;\mu_{gh}.\tau_g^2)}" title="p_{ih} = \frac{\pi_h \phi(y_i; \mu_{ih}.\tau_h^2)}{\sum_{g=1}^k \pi_g \phi(y_i; \mu_{gh}.\tau_g^2)}" /></a>

one can run an Expectation-Maximization algorithm to tune the parameters of the mix.

## Results

In the following a fit plot that compares the results obtained with the mixture of gaussians to the classical linear regression
model

![Output][image1]

## Instructions

It is assumed that an anaconda environment manager is installed on the local machine. 

1. Install the necessary Python packages.  

	For __Mac/OSX__  and __Linux__:
	```
		conda env create -f=./requirements.txt -n voleon-env
		source activate voleon-env
	```
	
	For __Windows__:
	```
		conda env create -f=./requirements.txt -n voleon-env
		activate voleon-env
	```
2. Pick a notebook.  

		jupiter notebook {NOTEBOOK}.ipynb

3. Enjoy!

## References

[1] Bartolucci, Francesco, and Luisa Scaccia. "The use of mixtures for dealing with non-normal regression errors." Computational Statistics & Data Analysis 48.4 (2005): 821-834


