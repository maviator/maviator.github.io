---
layout: post
title:  "How to set-up a Kaggle ready System in 3 steps"
date:   2017-12-10 16:03:47 +0100
categories: tutorial
---

It literally takes 5 min to set-up your system to be able to participate in Kaggle competitions. In this tutorial I will describe the steps to set up a python environment that can be used to compete in Kaggle competitions. This tutorial is for a Ubuntu OS.

## **Download Anaconda**

Anaconda is a python distribution for Data Science. This distribution has most of the tools needed to start practicing Data Science in a single package. This saves you the time and complications of installing each tool separately and worrying about dependencies. To download Anaconda for Ubuntu, visit their [download page](https://www.anaconda.com/download/#linux) and download the installer. There are two versions of python: python 2.7 and 3.6. I personally use the 2.7. The downloaded file should have a `.sh` extension.

## **Install Anaconda**

Open a **Terminal**, go to the downloaded file location and type the following command:

{% highlight bash %}
    bash Anaconda3-5.0.1-Linux-x86_64.sh
{% endhighlight %}

Note how the file name my be different if you have a 32-bit machine or if you chose python 2.7. To be safe, just type **Anaconda** and hit **Tab** and the right file name will be auto completed. This is a normal installation package, just hit yes and next whenever you have to.
These are the libraries that we are interested in that are installed now:

- Numpy
- Scipy
- Scikit learn
- Pandas
- Seaborn
- Matplotlib
- Jupyter

## **Install additional libraries**
 
Anaconda comes with plenty of libraries that we need already installed. However, we will add some additional libraries that are commonly used in Kaggle. We will install **XGBoost** and **LightGBM**. These are two Gradient Boosting libraries that are very common amongst kagglers.

{% highlight bash %}
    conda install -c conda-forge xgboost
{% endhighlight %}

This command will install XGBoost library through the Anaconda package manager

{% highlight bash %}
    conda install -c conda-forge lightgbm
{% endhighlight %}

This command will install LightGBM library through the Anaconda package manager

Now we are set to start competing in Kaggle!
