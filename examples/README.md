# Tutorial
This section provides tutorials to help you get started with FlexQuant. This package automates the generation of agents for flexibility quantification and follows the same structure as the [Agentlib-MPC](https://github.com/RWTH-EBC/AgentLib-MPC/tree/main/agentlib_mpc). For better understanding, it is recommended to refer to the [Agentlib-MPC documentation](https://rwth-ebc.github.io/AgentLib-MPC/main/docs/index.html). 
This section begins with the framework's structure, followed by the application. For more details, please refer to the [published paper](https://www.sciencedirect.com/science/article/pii/S0378778825002300).

## The framework

![Framework and data flow of the seven agents in FlexQuant](https://raw.githubusercontent.com/RWTH-EBC/AgentLib-FlexQuant/main/docs/images/FlexQuantFramework.jpg)
*Framework and data flow of the seven agents in FlexQuant*

In total, the framework consists of seven agents: Predictor Agent, BES Agent, three MPC Agents, an Indicator Agent and a market agent. The data exchange between these agents is illustrated with the arrows in the image above. The black boxes are a standard MPC setup created with [AgentLib](https://github.com/RWTH-EBC/AgentLib) and [Agentlib-MPC](https://github.com/RWTH-EBC/AgentLib-MPC/tree/main/agentlib_mpc). They serve as input for flexquant, the resulting output of which is represented with the grey boxes. For the normal use case without flexibility quantification, only the agents and communications in black are active. The ones in grey are generated while quantifying the flexibility. Detailed descriptions for each agent and their interactions can be found in this [section](#the-agents).

### The Agents

#### Predictor Agent
The Predictor Agent provides a prediction trajectory of the boundary
conditions for the given use case to the MPC Agents. This includes factors such as weather conditions, electricity tariffs, comfort boundaries, and occupancy schedules. The data can either be historical or retrieved via API services to support real-time operation.

#### BES Agent
The BES Agent simulates the energy system to be controlled. It can either use the same model as the MPC or a higher-fidelity one. In the latter case, the BES model does not need to be Python-based; for example, a Modelica model or even a real-world BES can be utilized. The BES Agent receives control signals from the MPC, applies them to the system, and subsequently sends the resulting measurements back to the MPC. 

#### MPC Agents
The key components of the FlexQuant framework are the three MPCs: the **baseline MPC**, which controls the BES and two **shadow MPCs** for the calculation of the available flexibility.  

The **Baseline MPC** is responsible for optimizing the operation of the BES with the objective of minimizing operational costs over the prediction horizon. While used for flexibility quantification, it is slightly modified to include the extra function of delivering the accepted flex offer.

The **Shadow MPCs** are designed to assess the maximum possible flexibility of electricity usage over a user-defined flexibility event duration. They are termed "shadow" because they do not directly control the BES but only support the evaluation of system flexibility. Two Shadow MPCs are employed: The Negative Shadow MPC calculates the control trajectory that maximizes BES power consumption, leading to a negative power contribution to the market (i.e., higher grid consumption).
 The Positive Shadow MPC does the opposite. The prediction horizon of the Shadow MPCs is divided as following:

![Split of the prediction horizon of the Shadow MPCs](https://raw.githubusercontent.com/RWTH-EBC/AgentLib-FlexQuant/main/docs/images/ShadowMPCTimeSlpit.jpg)
*Split of the prediction horizon of the Shadow MPCs*

The time t_MC is the market clearing time, during which a flexibility offer in t_FE is reserved and the market can decide whether to take it. The preparation time t_Prep allows the system to prepare itself for the upcoming flexibility event in advance to maximize the flexibility in t_FE, where the flexibility event takes place. 

Both the baseline and the shadow MPCs must have the storage variable ``E_stored`` for electrical energy as output, if the correction of the flexible energy cost is activated. According to definition in the package, ``E_stored`` increases as more electrical energy is stored in the system. Therefore, it should be defined as following:

- for heating case, E_stored = &sum; C * T / &eta; + other stored electrical energy
- for cooling case, E_stored = - &sum; C * T / &eta; + other stored electrical energy

where T is the temperature of the components in the system and &eta; could be e.g. the COP of a heat pump.

#### Indicator Agent
The Indicator Agent utilizes the power consumption predictions of the
three MPCs to calculate key performance indicators for quantifying available flexibility offers. They could be the total energy, the peak power, the average power or the cost etc.

#### Market Agent
Once the Market Agent decides to accept a flexibility offer, it sends the accepted flexibility trajectory back to the baseline MPC, which must deliver it in the corresponding time interval t<sub>FE</sub>.

## Application
This section demonstrates how to use the FlexQuant package. 

In general, a use case has the two following types of files:
- Flex_config: this is a json file that defines the configurations for the agents represented by the grey boxes in the [framework figure](#the-framework). It also specifies the modifications to the Baseline MPC when used in a FlexQuant framework compared to the standard control case. Note that not all the configurations are explicitly detailed within this file; instead, it may reference other configuration files, such as an indicator config in a separate JSON file.
- Modules: Each use case has its own specific BES, (Baseline) MPC and predictor module, represented as black boxes in the [framework figure](#the-framework). These Agents are implemented using the standard methods from  [AgentLib](https://github.com/RWTH-EBC/AgentLib) and [Agentlib-MPC](https://github.com/RWTH-EBC/AgentLib-MPC/tree/main/agentlib_mpc). For every module, there is a corresponding python file that defines its variables and functionality. Additionally, each module has a configuration JSON file, which can override the default variable values if specified.

The core functionality of this package is executed by calling:

 ``FlexAgentGenerator(flex_config, mpc_agent_config).generate_flex_agents()``

Below is a brief breakdown of how this method works.

### Breakdown
- ``__init__(flex_config, mpc_agent_config)`` initializes each field of the class with default value.
  1. load the flex_config
  2. initialize the three mpc agent config using the mpc_agent_config
  3. initialize the three mpc module config using the mpc agent config. All the mpc modules are now the same as the standard mpc, but will be later modified individually.
  4. do step 2 and 3 for indicator and market 
  

- ``generate_flex_agents()`` modifies the modules according to the flex_config. Things changed can be found in the adapt functions. 
  1. adapt mpc modules
  2. dump jsons of the mpc agents including the adapted module configs
  3. do step 1 and 2 for indicator and market 
  
The automatically generated configs and mpc models will be stored in the generated '**flex_files_directory**' folder and will be used during the simulation.
