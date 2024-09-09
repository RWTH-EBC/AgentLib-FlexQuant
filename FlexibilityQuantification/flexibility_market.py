import agentlib
from typing import List, Optional
import pandas as pd
from io import StringIO
import numpy as np
import os
from flex_offer import OfferStatus


class FlexibilityMarketConfig(agentlib.BaseModuleConfig):
    parameters: List[agentlib.AgentVariable] = [
        agentlib.AgentVariable(name="random_seed", description="random seed for reproducing experiments", value=None),
        agentlib.AgentVariable(name="pos_neg_rate", description="determines the likelihood positive and the negative flexibility"),
        agentlib.AgentVariable(name="offer_acceptance_rate", description="determines the likelihood of a accepted offer"),
        agentlib.AgentVariable(name="minimum_average_flex", description="minimum average of an accepted offer"),
        agentlib.AgentVariable(name="maximum_time_flex", description="maximum time flex of an accepted offer"),
        agentlib.AgentVariable(name="time_step", description="timestep of the MPC"),
        agentlib.AgentVariable(name="cooldown", value=6, description="cooldown time (no timesteps) after a provision"),
        agentlib.AgentVariable(name="forced_offers")


    ]
    inputs: List[agentlib.AgentVariable] = [
        # agentlib.AgentVariable(name="PowerFlexibilityOffer")
    ]
    outputs: List[agentlib.AgentVariable] = [
        agentlib.AgentVariable(
            name="_P_external", alias="_P_external",
            description="External Power IO"
        ),
        agentlib.AgentVariable(
            name="rel_start", alias="rel_start",
            description="relative start time of the flexibility event"
        ),
        agentlib.AgentVariable(
            name="rel_end", alias="rel_end",
            description="relative end time of the flexibility event"
        ),
        agentlib.AgentVariable(
            name="in_provision", alias="in_provision",
            description="Set if the system is in provision", value=False
        )
    ]
    shared_variable_fields:List[str] = ["outputs"]


class FlexibilityMarket(agentlib.BaseModule):
    config: FlexibilityMarketConfig
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results_file: Optional[str] = kwargs.get("results_file") or "flexibility_market_results.csv"
        self.df = None
        self.end = 0
        for parameter in self.config.parameters:
            if parameter.name == "random_seed":
                self.set_random_seed(parameter.value)
            elif parameter.name == "pos_neg_rate":
                self.pos_neg_rate = parameter.value
            elif parameter.name == "offer_acceptance_rate":
                self.offer_acceptance_rate = parameter.value
            elif parameter.name == "minimum_average_flex":
                self.minimum_average_flex = parameter.value
            elif parameter.name == "maximum_time_flex":
                self.maximum_time_flex = parameter.value

    def set_random_seed(self, random_seed):
        """set the random seed for reproducability"""
        self.random_generator = np.random.default_rng(seed=random_seed)

    def get_results(self) -> Optional[pd.DataFrame]:
        """
        Opens results file of flexibilityindicators.py
        results_file defined in __init__
        """
        results_file = self.results_file
        try:
            results = pd.read_csv(results_file, header=[0], index_col=[0,1])
            return results
        except FileNotFoundError:
            self.logger.error("Results file %s was not found.", results_file)
            return None
            
    def register_callbacks(self):
        self.agent.data_broker.register_callback(
            name="PowerFlexibilityOffer", alias="PowerFlexibilityOffer", callback=self.flexibility_callback
        )
        self.df = None
        self.cooldown_ticker = 0


    def write_results(self, offer):
        if self.df is None:
            self.df = pd.DataFrame()
        df = offer.dataframe()
        index_first_level = [self.env.now] * len(df.index)
        multi_index = pd.MultiIndex.from_tuples(zip(index_first_level, df.index))
        self.df = pd.concat((self.df, df.set_index(multi_index)))
        indices = pd.MultiIndex.from_tuples(self.df.index, names=["time_step", "time"])
        self.df.set_index(indices, inplace=True)
        self.df.to_csv(self.results_file)

    def flexibility_callback(self, inp, name):
        """
        When a flexibility offer is sent this function is called. 
        
            The offer is accepted randomly. The factor self.offer_acceptance_rate determines the
                random factor for offer acceptance. self.pos_neg_rate is the random factor for
                the direction of the flexibility. A higher rate means that more positive offers will be accepted.
            Forced offers: if set, than an offer is accepted without the randomness and constraints.
            
            Constraints:
                cooldown: during $cooldown steps after a flexibility event no offer is accepted
                minimum_average_flex: min amount of flexibility to be accepted, to account for the model error
                maximum_time_flex: to be accepted, flexibility must be ready as soon as possible. If most of the
                    flexibility is only available at the end of the flexibility event, dont accept the offer!
        """
        
        offer =  inp.value
        # check if there is a flexibility provision and the cooldown is finished
        if str(self.env.time) in self.get("forced_offers").value.keys():
            forced, power_multiplier = self.get("forced_offers").value[str(self.env.time)]
        else: 
            forced = None
            power_multiplier = 1#np.random.choice([0.4,0.5,0.6,0.7,0.8,0.9,1], p=[0.01, 0.01, 0.05, 0.05, 0.1, 0.1, 0.68])
        if not self.get("in_provision").value and self.cooldown_ticker == 0:
            if forced is not None or self.random_generator.random() < self.offer_acceptance_rate:  
                profile = None
                if forced == "positive" or np.average(offer.pos_diff_profile) > self.minimum_average_flex:
                    if forced == "positive"  or self.random_generator.random() < self.pos_neg_rate:
                        if forced == "positive" or offer.pos_time_flex < self.maximum_time_flex:
                            profile = offer.base_power_profile - offer.pos_diff_profile * power_multiplier
                            offer.status = OfferStatus.accepted_positive
                
                if forced == "negative" or (profile is None and np.average(offer.neg_diff_profile) > self.minimum_average_flex):
                    if forced == "negative" or offer.neg_time_flex < self.maximum_time_flex:
                        profile = offer.base_power_profile + offer.neg_diff_profile * power_multiplier
                        offer.status = OfferStatus.accepted_negative

                if profile is not None:
                    offer.power_multiplier = power_multiplier
                    profile = profile.dropna()
                    profile.index += self.env.time
                    self.set("_P_external", profile)
                    self.end = profile.index[-1]
                    self.set("in_provision", True)    
                    self.cooldown_ticker = self.get("cooldown").value


        elif self.cooldown_ticker > 0:
            self.cooldown_ticker -= 1
        
        self.write_results(offer)
        

    def cleanup_results(self):
        results_file = self.results_file
        if not results_file:
            return
        os.remove(results_file)

    def process(self):
        while True:
            # End the provision at the appropriate time
            if self.end < self.env.time:
                self.set("in_provision", False)
            yield self.env.timeout(self.env.config.t_sample)
       