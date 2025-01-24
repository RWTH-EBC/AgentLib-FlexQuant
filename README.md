# Flexibility Quantification

This project is a plugin for the [AgentLib](https://github.com/RWTH-EBC/AgentLib). This agent-based framework employs model predictive control (MPC) to quantify flexibility offers of electricity usage of building energy systems (BES) during operation.

## Installation
To install, you can either use the ``requirements.txt`` or go for package installation with ``pip install -e .``. 
The ``-e`` option installs the package in editable mode, which should be done when working on this package.
 This project is compatible with Python 3.12 (3.11, 3.10 to be checked).

## Author
- Felix Stegemerten 

## Referencing the FlexQuant
A publication regarding the FlexQuant is currently in the work. A preprint is available under https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5015569

## Tutorial
This section provides tutorials to help you get started with FlexQuant. It begins with an introduction to the framework's structure, followed by a detailed breakdown of an example to guide you through its application.

### The framework

<figure>
  <img src="./docs/images/FlexQuantFramework.jpg" width="600" alt="framework">
  <figcaption>Framework and data flow of the seven agents in FlexQuant</figcaption>
</figure>


In total, the framework consists of seven agents: Predictor Agent, BES Agent, three MPC Agents, an Indicator Agent and a market agent. The data exchange between these agents is illustrated with the arrows in the image above. For the normal use case without flexibility quantification, only the agents in the black box are active. The ones in the grey box are generated while quantifying the flexibility, Let’s take a closer look at each agent and their interactions.

<ins>Predictor Agent</ins> \
The Predictor Agent provides a prediction trajectory of the boundary
conditions for the given use case to the MPC Agents. This includes factors such as weather conditions, electricity tariffs, comfort boundaries, and occupancy schedules. The data can either be historical or retrieved via API services to support real-time operation.

<ins>BES Agent</ins> \
The BES Agent simulates the energy system to be controlled. It can either use the same model as the MPC or a higher-fidelity model. In the latter case, the BES model does not need to be Python-based; for example, a Modelica model or even a real-world BES can be utilized. The BES Agent receives control signals from the MPC, applies them to the system, and subsequently sends the resulting measurements back to the MPC. 

<ins>MPC Agents</ins> \
The key components of the FlexQuant framework are the three MPCs: the **baseline MPC**, which controls the BES and two **shadow MPCs** for the calculation of the available flexibility.  

The **Baseline MPC** is responsible for optimizing the operation of the BES with the primary objective of minimizing operational costs over the prediction horizon. Notably, only the control actions determined by the Baseline MPC are actually applied to the BES.

The **Shadow MPCs** are designed to assess the maximum possible flexibility in electricity usage over a user-defined flexibility event duration. These controllers are termed "shadow" because they do not directly control the BES; instead, they support the evaluation of system flexibility. Two Shadow MPCs are employed: The Negative Shadow MPC calculates the control trajectory that maximizes BES power consumption, leading to a negative power contribution to the market (i.e., higher grid consumption).
 The Positive Shadow MPC does the opposite. The horizon of the Shadow MPCs is divided as following: 

<figure>
  <img src="./docs/images/ShadowMPCTimeSlpit.jpg" width="600" alt="framework">
  <figcaption>Split of the prediction horizon of the Shadow MPCs</figcaption>
</figure>

The time t<sub>MC</sub> is the market clearing time, during which a flexibility offer is active and the market can decide whether to take it. t<sub>Prep</sub> is the preparation time, where the system can prepare itself for the upcoming flexibility event in advance. In t<sub>FE</sub> the flexibility event takes place. 

<ins>Indicator Agent</ins> \
The Indicator Agent utilizes the power consumption predictions of the
three MPCs to calculate indicators for quantifying available flexibility offers. It could be the total energy, the peak power, the average power or the cost etc.

<ins>Market Agent</ins> \
Once the Market Agent decides to accept a flexibility offer, it sends the accepted flexibility trajectory back to the baseline MPC, and it must deliver it in the corresponding time interval t<sub>FE</sub>.

### Example
This section demonstrates how to use the FlexQuant package. Examples can be found in the folder [Examples](Examples). 

In general, a use case has the following files:
- Flex_config: it is the json file that specifies the configurations for the agents in the grey boxes in the [framework](#the-framework). Additionally, it also includes the change to the Baseline MPC when used in a FlexQuant framework compared to the normal control case. Note that not all the configs are listed in detail in this file. Instead, it may refer to e.g. an indicator config in a separate json file.
- Models: every use case has its own BES, (Baseline) MPC and predictor model. These are the black boxes in the [framework](#the-framework). For every model there exists a python file, where the variables and functionality are defined. Also, there is a config json file for each. The default value of the variables will be replaced with the one in the config, if provided. 

To see how the package generates and updates everything, read more [here](flexibility_quantification/README.md)