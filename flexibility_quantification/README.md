
### Introduction
This package automates the generation of agents for flexibility quantification and follows the same structure as the [Agentlib-MPC](https://github.com/RWTH-EBC/AgentLib-MPC/tree/main/agentlib_mpc). For better understanding, it is recommended to refer to the [Agentlib-MPC documentation](https://rwth-ebc.github.io/AgentLib-MPC/main/docs/index.html).

The core functionality this package is executed by calling:

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
  
The automatically generated configs and mpc models will be stored in the generated '**created_flex_files**' folder and will be used during the simulation.
