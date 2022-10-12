# Predictive control of solar thermal plants through the application of Deep Reinforcement Learning techniques

This repository contains the results obtained in the Master Thesis of the M.Sc. in Industrial Engineering (University of Seville, Spain) of Fernando Borrego. This thesis, entitled "Predictive control of solar thermal plants through the application of Deep Reinforcement Learning techniques", explores the opportunities available in the field of process control for Deep Reinforcement Learning techniques through a practical application to the case of solar thermal plants.

The aforementioned project can be found at: (... on hold ...)

## Abstract

Renewable generation technologies have experienced an unprecedented growth in popularity over the last few decades. This growth, driven by lower costs and a need to reduce the use of those technologies that cause -among others- climate change, has encouraged the study of concentrating solar power technology as an appealing alternative given its energy storage capacity. 

The control of this type of installations is often based on Model Predictive Control (MPC) strategies. However, due to the excessive computational cost of needing to solve an optimization problem at each sampling time and the need for such a precise knowledge of the dynamics of the system to be controlled, this control strategy can sometimes be inefficient. 

This work, therefore, aims to present an alternative control strategy to these MPC controllers for controlling parabolic trough collector solar plants (ACUREX plant). To this end, a control strategy based on reinforced learning techniques is proposed in order to not only reproduce the good results provided by a predictive controller, but also to solve the problems arising from it. Thus, it is developed a control agent which, by means of a DDPG learning algorithm, shapes an optimal control law that maximizes the thermal power produced by the plant. This algorithm is further complemented with the development of an ACUREX simulator using the library _OpenAI Gym_, which allows the results achieved to be checked.

## Results

( ... WIP ...)


## File Structure

The file structure of the current project is the following:

    |-- README.md
    |-- LICENSE
    |-- requirements.txt
    |-- data
        |-- real
        |-- test
    |-- figures
    |-- log
    |-- saves
    |-- src
        |-- agents
            |-- DDPGAgent.py
            |-- MPCController.py
        |-- environments
            |-- ACUREXEnv.py
        |-- models
            |-- ConcentratedModel.py
        |-- networks
            |-- temp
            |-- DDPGNewtorks.py
        |-- utils
            |-- buffer.py
            |-- clipper.py
            |-- graphics.py
            |-- noise.py
        |-- control.py
        |-- train.py
        |-- evaluate.py









- - -
October, 2022. Fernando Borrego Prado.