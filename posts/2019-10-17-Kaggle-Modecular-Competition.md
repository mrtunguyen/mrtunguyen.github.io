---
aliases:
- /kaggle/2019/10/17/Kaggle-Modecular-Competition
categories:
- kaggle
- graph neural network 
date: '2019-10-17'
description: Tricks and lessons I have learned from getting into top 5 of Kaggle Molecular
  Competition
layout: post
title: 'Kaggle Molecular Competition: Lessons Learned from Finishing in Top 5'
toc: true
image: assets/kaggle.png

---

In the recent [Kaggle Predicting Molecular Properties Competition](https://www.kaggle.com/c/champs-scalar-coupling), my team has managed to finish in 5th place (out of 2749 teams). It is the first competition that I spent seriously a lot of time and I learned a lot through it. Though I don't consider myself as a Kaggle expert by any means, I want to share some lessons, insights that hopefully can be helpful for others.

# A little words about Kaggle 
## Kaggle is the best school, participants are the best teachers and my teammates are the best companions I've had

Kaggle is undoubtly a great platform with all sorts of interesting problems along with datasets, kernels and great discussions. Like what I saw in a post long time ago, Kaggle is definitely a home for data science enthusiats all around the world where they spend their days and nights to challenge themselves. And now I would say that I'm very proud to be one of them. Since I'm not graduated from any school formation of data science like many of others, Kaggle comes to me as a place where I can learn many things, keep me motivated in the field which is moving a lot every day. The people like [Heng](https://www.kaggle.com/hengck23) are the best teachers I've had, not only from their insights and sharing about competitions but also from the way how they work.
Moreover, one of the most important features I love about Kaggle is Leaderboard where we could see where we're standing compared to others. During this competition, I admit that the first thing I did when waking up every morning is looking at the leaderboard. Seeing many competitors passing above me in the ranking helps me to benchmark my skills, push me to learn and try new things which is very important in my career.
So if you want to really get into Machine learning or Data Science, I believe that Kaggle comes in as one of the best ways.


# About the competition: Predicting Molecular Properties

This competition is sponsored by the the Chemistry and Mathematics in Phase Space (CHAMPS) at the University of Bristol, Cardiff University, Imperial College and the University of Leeds. It requires competitors to predict the magnetic interaction between two atoms in a molecule (i.e the scalar coupling constant). The objective of using machine learning models in this kind of project is to allow medicinal chemists to gain structural insights faster and cheaper than the quantum mechanic methods and to enable scientists to understand how the 3D chemical structure of a molecule affects its propeties and behavior. Such tools will enable researchers to make progress in a range of important problems, like designing molecules to carry out specific cellular tasks, or designing better drug molecules to fight disease. See more about it in [here](https://www.kaggle.com/c/champs-scalar-coupling)

# Background

At first, my team didn't had any domain expertise, prior knowledge about chemistry or molecular properties. But we found this competition very interesting to discover a new application of machine learning. [Lam](https://www.kaggle.com/lamdang) is the first who started in our team and realized that using Graph Neural Network (GNN) gains better score than classical machine learning models (like Xgboost or LightGBM). He found out an interesting [paper](https://arxiv.org/abs/1812.05055) about using GNN for molecular properties and tried to implement it in this competition. After few submissions, he managed to get into top 3 in public Leaderboard at the beginning of the competition. [Guillaume](https://www.kaggle.com/grjhuard) and I joined it later with different angle of view. Firstly, I tried to experiment ideas of Heng, using [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212) for which he published in the dicussion forum (with his starter code). I also devoted an amount of my time to grasp some knowledge about molecular properties and their interaction. Thereafter, I read different relevant papers in this kind of task in order to better understand how GNN works. The interesting idea behind GNN from machine learning standpoint is finding a representation of a molecule as a graph with atoms as nodes and bonds as edges and how information flow between nodes-nodes, nodes-bonds or bonds-bonds. For example, the assumption that two nodes have a relationship and interact each other is expressed by an edge between them. Similarly, the absence of an edge expresses the assumption that the nodes have no relationship and should not influence each other directly. I also tried another deep learning architecture like [Schnet](https://arxiv.org/pdf/1806.01261.pdf) or adding another molecular features by reading dicussions/kernels in kaggle forum before sticking into [MEGNET](https://arxiv.org/abs/1812.05055). Lam, Guillaume and I decided to merger as a team one month before competition end when each person has his own good competitive model. For more details, our solution write-up can be found [here](https://www.kaggle.com/c/champs-scalar-coupling/discussion/106864#614330)

And now, below are few lessons that I've learned so far from this competition: 

## 1. A good validation strategy is half of success
I read an [interview](http://blog.kaggle.com/2018/05/07/profiling-top-kagglers-bestfitting-currently-1-in-the-world/) of bestfitting about his strategy of winning competitions in Kaggle. He said that a good CV is half of success. I couldn't agree more. Every try we made, it should improve both on our local CV and on the public LB. In this competition, we set aside 5000 moleculars for validation and 80000 ones for training. And luckily, validation score and public leaderboard score is very close so that we are very sure about evaluation of our models. This closeness makes us feel confident with our stacking strategy at the end.    
So, always trust your CV more than the leaderboard. The public leaderboard represents only 29% of the actual test set, so you can't be sure about the quality of your solution based on this percentage. Sometimes your model might be great overall, but bad on the data, specially in the public test set.
The experience from [here](https://www.linkedin.com/pulse/my-team-won-20000-1st-place-kaggles-earthquake-corey-levinson/) and [recent finished competition](https://www.kaggle.com/c/severstal-steel-defect-detection) make this lesson more valuable. 

## 2. Classical Machine learning ideas possibly work in deep learning
When using classical machine learning models like xgboost or Lightgbm, we often heard many times about feature importance technique. While they are widely used in many tabular problems, it is less likely happen in deep learning. But my teammate Guillaume has proven the opposite. After testing feature importance (by randomize one feature at the prediction step), he noticed that the most important feature was by far the angle between an edge and the edge with the closest atom to the first edge. This insight gave us a 0.15 improvement for our best single model.

## 3. Always keep updated state-of-the-art of the field (either NLP or Computer Vision)
I encountered many data scientists who said that since they are only working in Computer Vision, they don't have any interest to invest their time in NLP models. For me, I don't feel that way. When this competition was finished and the solution of top teams were released, all top 6 teams, except our team, were using a technique that is recently very popular in NLP community - [Transformer](https://arxiv.org/abs/1706.03762). The way that they integrated Transformer in their model is quite eye opening for us. 

## 4. Test ideas with simple model before going bigger 
One of the big mistakes I made during this competition is implementing a quite big models in first try. In fact, when my teammates told me about megnet models, I read the paper, write code from scatch and run it with 5 layers while trying to add some new ideas. It took me half day to run and realized that it doesn't converge at all. Since it is quite deep, I'm kind of stuck in finding why this model doesn't work as expected. After discussing with my team and simplifing model to only 1 layer, I figured out errors in my implementation. One of the important errors is using Normalization. Indeed, BatchNorm makes the loss fluctuate a lot while LayerNorm works much better in this case. 

## 5. How to structure code when testing many ideas
One of the problems when working in projects like Kaggle competitions is that how we can make plan of our code so that we can reproduce the results of previous ideas (even idea we tested one month ago) whenever we want. The problem will get bigger and bigger when we're trying more sophisticated way that make taking notes difficult. The lesson I've learned from [Heng](https://www.kaggle.com/hengck23) when looking at his starter code is that create first folder of dataset which contains all dataset, seconde folder which has common functions and a third folder for my principal code. When I want to test new different idea, I will make a copy of the third model and make changes on it. Following this way can make project management easier and I can reproduce my results whenever I need.

## 6. Working in a team helps you go faster and further
Besides learning the technical parts of the competition, a very important thing I've learned was how to work in a team. We all have work during the week, so we can only do Kaggle at our free time and have to do it in a smart way. Luckily, we worked at the same place, communicated every day and bounced ideas off each other. This competition is my first gold medal! I'm very glad I had this experience and very thankful to be a part of a wonderful team with Lam and Guillaume. I hope we will have more opportunities to work together in future competitions! 

