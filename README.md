# Predictive control of solar thermal plants through the application of Deep Reinforcement Learning techniques

This repository contains the results obtained in the Master Thesis of the M.Sc. in Industrial Engineering (University of Seville, Spain) of Fernando Borrego. This thesis, entitled "Predictive control of solar thermal plants through the application of Deep Reinforcement Learning techniques", explores the opportunities available in the field of process control for Deep Reinforcement Learning techniques through a practical application to the case of solar thermal plants.

The aforementioned project can be found at: (... on hold ...)

## Abstract

Renewable generation technologies have experienced an unprecedented growth in popularity over the last few decades. This growth, driven by lower costs and a need to reduce the use of those technologies that cause -among others- climate change, has encouraged the study of concentrating solar power technology as an appealing alternative given its energy storage capacity. 

The control of this type of installations is often based on Model Predictive Control (MPC) strategies. However, due to the excessive computational cost of needing to solve an optimization problem at each sampling time and the need for such a precise knowledge of the dynamics of the system to be controlled, this control strategy can sometimes be inefficient. 

This work, therefore, aims to present an alternative control strategy to these MPC controllers for controlling parabolic trough collector solar plants (ACUREX plant). To this end, a control strategy based on reinforced learning techniques is proposed in order to not only reproduce the good results provided by a predictive controller, but also to solve the problems arising from it. Thus, it is developed a control agent which, by means of a DDPG learning algorithm, shapes an optimal control law that maximizes the thermal power produced by the plant. This algorithm is further complemented with the development of an ACUREX simulator using the library _OpenAI Gym_, which allows the results achieved to be checked.

## Results
It is interesting to use, as a way of quantifying the performance of the DRL agent against the one shown by the MPC, different evaluation metrics:

| Metric  | MPC     | DRL     | (DRL - MPC) / MPC |
|---------|---------|---------|-------------------|
| Score   | 563.204 | 563.248 | +0.01%            |
| Time    | 2.245   | 0.025   | -98.89%           |
| Power   | 111.244 | 109.466 | -1.60%            |
| AACI    | 0.371   | 0.378   | +1.89%            |
| MSCV    | 61.352  | 45.113  | -26.47%           |


Despite the fact that the reinforcement learning agent achieves an average thermal power slightly lower than the one achieved by the predictive controller, it can be concluded from these metrics that the control agent developed in this work is a valid alternative for predictive control in solar thermal plants. Thus, it can be noticed how the accumulated score (metric used as an indicator of the quality of the solution) for both methods are practically the same.

Moreover, it has been proven that reinforced learning is truly ahead of predictive control techniques in terms of execution speed. In this way, DRL controllers can save up to 99% in execution time and, consequently, in hardware resources used. This is a benefit when exploiting this type of controller, since the time taken by non-linear predictive controllers often makes it unfeasible to use them for real plant control.

Finally, with regard to compliance with the restrictions, it is interesting to make the following comments:
* Firstly, very similar results can be noticed in the cumulative slew rate during the simulation. These results indicate that both controllers achieve a mostly smooth response.

* In second place, greater differences are observed with respect to the MSCV. In this case, DRL algorithm is able to offer better results than MPC controller, reaching up to 25% less penalty.

In conclusion, it has been demonstrated that the DRL agent proposed in this work has not only provided results comparable to those achieved with the predictive controller, but has also significantly reduced the execution time required for this purpose.


## References

#### ACUREX model:

[1] Camacho, E. F., Berenguel, M., and Rubio, F. R. Advanced control of solar plants.

[2] Berenguel, M., Camacho, E. F., and Rubio, F. R. Control of solar energy systems. Springer, 2012.

[3] Camacho, E. F., Rubio, F., and Gutierrez, J. Modelling and simulation of a solar power plant with a distributed collectors system. In Power Systems: Modelling and Control Applications. Elsevier, 1989, pp. 321–325.

[4] Carmona Contreras, R. Análisis, modelado y control de un campo de colectores solares distribuidos con sistema de seguimiento en un eje.

[5] Al-Naima, F., and Abdul Majeed, B. Spline-based formulas for the determination of equation of time and declination angle. International Scholarly Research Notices 2011 (2011).

[6] Dudley, V., and Workhoven, R. Performance testing of the acurex solar-collector model 3001-03. Tech. rep., Sandia National Lab.(SNL-NM), Albuquerque, NM (United States), 1982.

[7] Ruiz-Moreno, S., Frejo, J. R. D., and Camacho, E. F. Model predictive control based on deep learning for solar parabolic-trough plants. Renewable Energy 180 (2021), 193–202.


#### DDPG:

[8] Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., and Riedmiller, M. Deterministic policy
gradient algorithms. In International conference on machine learning (2014), PMLR, pp. 387–395.


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