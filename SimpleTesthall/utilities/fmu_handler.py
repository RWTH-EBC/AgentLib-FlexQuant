#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Handles all interactions with a FMU simulation

Author: aku
"""

import fmpy
import shutil
import time

class fmu_handler():
    """
    The fmu handler class
    """

    def __init__(self,
            start_time,
            stop_time,
            step_size,
            sim_tolerance,
            fmu_file,
            instanceName,
            ):
        """[summary]

        Args:
            start_time ([int]): Start time in sec usually 0 
            sim_date ([datetime]): the datetime of the simulation time
            stop_time ([int]): Stop time in sec 
            step_size ([int]): The time step size after which data is exchanged
            sim_tolerance ([float]): The numeric tolerance of the solver usual 0.001
            fmu_file ([string]): The name of the FMU file
            instanceName ([type]): A name of the FMU instance. FMPY specific can be random
        """
        self.start_time = start_time  # start time
        self.stop_time = stop_time  # stop time
        self.step_size = step_size  # The macro time step
        self.sim_tolerance = sim_tolerance  # The total simulation tolerance
        self.fmu_file = fmu_file
        self.instanceName = instanceName


        # read the model description
        self.model_description = fmpy.read_model_description(self.fmu_file)


        # Collect all variables
        self.variables = {}
        for variable in self.model_description.modelVariables:
            self.variables[variable.name] = variable

        # extract the FMU
        self.unzipdir = fmpy.extract(self.fmu_file)

        # create fmu obj
        self.fmu = fmpy.fmi2.FMU2Slave(guid=self.model_description.guid,
                                       unzipDirectory=self.unzipdir,
                                       modelIdentifier=self.model_description.coSimulation.modelIdentifier,
                                       instanceName=self.instanceName)


        #instantiate fmu
        self.fmu.instantiate()
        


    def setup(self):       
        # The current simulation time
        self.current_time = self.start_time
        
        # initialize model
        self.fmu.reset()
        self.fmu.setupExperiment(
            startTime=self.start_time, stopTime=self.stop_time, tolerance=self.sim_tolerance)


        
    def initialize(self):    
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()        
        
        
        
    def find_vars(self, start_str:str):
        """
        Retruns all variables starting with start_str
        """
        key = list(self.variables.keys())
        key_list=[]
        for i in range(len(key)):
            if key[i].startswith(start_str):
                key_list.append(key[i])
        return key_list



    def get_value(self, var_name:str):
        """
        Get a single variable.
        """
    
        variable = self.variables[var_name]
        vr = [variable.valueReference]
    
        if variable.type == 'Real':
            return self.fmu.getReal(vr)[0]
        elif variable.type in ['Integer', 'Enumeration']:
            return self.fmu.getInteger(vr)[0]
        elif variable.type == 'Boolean':
            value = self.fmu.getBoolean(vr)[0]
            return value != 0
        else:
            raise Exception("Unsupported type: %s" % variable.type)



    def set_value(self, var_name, value):
        """
        Set a single variable.
        var_name: str
        """
    
        variable = self.variables[var_name]
        vr = [variable.valueReference]
    
        if variable.type == 'Real':
            self.fmu.setReal(vr, [float(value)])
        elif variable.type in ['Integer', 'Enumeration']:
            self.fmu.setInteger(vr, [int(value)])
        elif variable.type == 'Boolean':        
            self.fmu.setBoolean(vr, [value == 1.0 or value == True or value == "True"])
        else:
            raise Exception("Unsupported type: %s" % variable.type)



    def do_step(self):     
        #check if stop time is reached
        if self.current_time < self.stop_time:
            #do simulation step
            status = self.fmu.doStep(
                currentCommunicationPoint=self.current_time,
                communicationStepSize=self.step_size)
            #augment current time step
            self.current_time += self.step_size 
            finished = False    
        else:
            print('Simulation finished')
            finished = True
            
        return finished



    def close(self):
        self.fmu.terminate()
        self.fmu.freeInstance()
        shutil.rmtree(self.unzipdir)
        print('FMU released')



    def read_variables(self, vrs_list:list):
        """ 
        Reads multiple variable values of FMU.
        vrs_list as list of strings
        Method retruns a dict with FMU variable names as key
        """
        res = {}
        #read current variable values ans store in dict
        for var in vrs_list:
            res[var] = self.get_value(var)
            
        #add current time to results   
        res['SimTime'] = self.current_time   
        
        return res



    def set_variables(self, var_dict:dict):
        '''
        Sets multiple variables.
        var_dict is a dict with variable names in keys.
        '''
        
        for key in var_dict:        
            self.set_value(key, var_dict[key])
        return "Variable set!!"



    def __enter__(self):
        self.fmu.terminate()
        self.fmu.freeInstance()
        
        
        
        
if __name__ == "__main__":
    fmu = fmu_handler(start_time=0,
            stop_time=3600*24*10,
            step_size=10,
            sim_tolerance=0.0001,
            fmu_file='EONERC_MainBuildingWithController.fmu',
            instanceName='test1',
            )    
    
    #read variables for output
    variables = fmu.find_vars('bus')  
    
    fmu.setup()
    #set variables
#    fmu.set_variables({'volumeFlow_air':3000, 'T_Air_In':10, 'T_water_in':70, 
#                       'T_set':20, 'registerBus1.hydraulicBus.valSet':0.6, 'registerBus1.hydraulicBus.pumpBus.rpm_Input':3000,
#                       'registerModule.T_start':280}) 
    
    fmu.initialize()
    finished = False
    t1 = time.time()
    results = []
    while not finished: 
        res = fmu.read_variables(variables) 
        print(fmu.current_time)    
        finished = fmu.do_step()
        results.append(res)
#   
    print('Time: ' + str(time.time()-t1))
#    fmu.close()    
        

       