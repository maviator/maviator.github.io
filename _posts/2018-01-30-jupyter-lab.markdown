---
layout: post
title:  "Jupyter Lab"
date:   2018-01-30 9:03:47 +0100
categories: other
tags: MLtopics jupyter-notebook
---

{% include image.html
            img="assets/jupyter_lab_cover.png"
            title="coursera"
            %}


Jupyter notebooks are the default medium data scientist use for their work and it's for a good reason. The ability to explore the data, run commands, plot graphs and have all of these features in one document is the major reason for the widespread use of Jupyter notebooks. However, there are other features that would be nice to have to better streamline your work and this is where the guys at Jupyter introduced [Jupyter Lab](https://github.com/jupyterlab/jupyterlab). It's a full featured IDE that has everything we ever wanted to be in Jupyter notebooks.

In this post, I will go through some of the main features of Jupyter Lab.

## Drag and Drop

Drag and drop is probably the best new feature that is added to jupyter notebooks. The ability to re order cells without cut and paste is powerful. It also feels more natural to do drag and drop given that code is organized in cells in notebooks.

{% include image.html
            img="assets/gifs/drag_drop.gif"
            title="coursera"
            %}

## Multiple notebooks split

Running multiple notebooks at the same time already exist with the jupyter notebooks. However, these notebooks had to be oppened in multiple browser windows. In jupyter Lab, you can have multiple notebooks open at the same time and in the same browser window. Also, you can arrange your notebooks as you like which gives more flexibility. Another nice feature is it's possible to have each notebook running on it's own kernel, this is powerful when running multiple notebooks at the same time doing different things.

{% include image.html
            img="assets/gifs/2windows.gif"
            title="coursera"
            %}

## Real time markdown editor

This is my personal favorite feature. Having my [blog](https://maviator.github.io/) posts in markdown format, I used to export notebooks in markdown format and then use Atom to edit the exported file, then run a local server to see the output of my edits. With this new feature, I can edit and see in real time the update of my markdown files in jupyter Lab. This speeds-up the edit process and streamlines work.

{% include image.html
            img="assets/gifs/md_editor.gif"
            title="coursera"
            %}

## Multiple windows

This is another favorite feature. Having to sync my work with my Github, I usually have to open another terminal and then commit the changes that I have made. With multiple windows open at the same time, I can have multiple notebooks that I am working on and then use a terminal inside jupyter Lab and commit my work. Again, awesome for streamlining work.

{% include image.html
            img="assets/gifs/mlti_window.gif"
            title="coursera"
            %}

## Other features

This is a list of some other features in jupyter Lab:
- Full file explorer
- Manage kernels and terminals
- Command search
- Fast CSV files viewer
- Real time collaboration on files hosted on Google Drive

{% include image.html
            img="assets/gifs/tabs.gif"
            title="coursera"
            %}

## Demo and presentation

<iframe width="560" height="315" src="https://www.youtube.com/embed/dSjvK-Z3o3U" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
