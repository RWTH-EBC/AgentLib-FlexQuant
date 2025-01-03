import agentlib
from typing import List, Optional, Union
import pandas as pd
import numpy as np
import os
from flexibility_quantification.data_structures.flex_offer import OfferStatus
import pydantic
from pathlib import Path
from flexibility_quantification.data_structures.market import MarketSpecifications


class FlexibilityMarketModuleConfig(agentlib.BaseModuleConfig):
    # parameters: List[agentlib.AgentVariable] = [
    # ]
    inputs: List[agentlib.AgentVariable] = [
        agentlib.AgentVariable(name="FlexibilityOffer")
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

    market_specs: MarketSpecifications
    # TODO: time_step needed?
    time_step: int = pydantic.Field(name="time_step", default=None, description="time step of the MPC")

    results_file: Optional[Path] = pydantic.Field(default=None)

    save_results: Optional[bool] = pydantic.Field(validate_default=True, default=None)
    overwrite_result_file: Optional[bool] = pydantic.Field(default=False, validate_default=True)

    shared_variable_fields: List[str] = ["outputs"]


class FlexibilityMarketModule(agentlib.BaseModule):
    """Class to emulate flexibility market. Receives flex offers and accepts these.

    """
    config: FlexibilityMarketModuleConfig

    # TODO: add docu
    df: pd.DataFrame = None
    end: Union[int, float] = 0

    def set_random_seed(self, random_seed):
        """set the random seed for reproducability"""
        self.random_generator = np.random.default_rng(seed=random_seed)

    def get_results(self) -> Optional[pd.DataFrame]:
        """
        Opens results file of flexibilityindicators.py
        results_file defined in __init__
        """
        results_file = self.config.results_file
        try:
            results = pd.read_csv(results_file, header=[0], index_col=[0, 1])
            return results
        except FileNotFoundError:
            self.logger.error("Results file %s was not found.", results_file)
            return None

    def register_callbacks(self):
        if self.config.market_specs.type == "custom":
            callback_function = self.custom_flexibility_callback
        elif self.config.market_specs.type == "single":
            callback_function = self.single_flexibility_callback
        elif self.config.market_specs.type == "random":
            callback_function = self.random_flexibility_callback
            self.set_random_seed(self.config.market_specs.options.random_seed)
        else:
            self.logger.error("No market type defined. Available market types are single, random "
                              "and custom. Code will proceed without market interaction.")
            callback_function = self.dummy_callback

        self.agent.data_broker.register_callback(
            name="FlexibilityOffer", alias="FlexibilityOffer",
            callback=callback_function
        )

        self.df = None
        self.cooldown_ticker = 0

    def write_results(self, offer):
        if self.df is None:
            self.df = pd.DataFrame()
        df = offer.as_dataframe()
        index_first_level = [self.env.now] * len(df.index)
        multi_index = pd.MultiIndex.from_tuples(zip(index_first_level, df.index))
        self.df = pd.concat((self.df, df.set_index(multi_index)))
        indices = pd.MultiIndex.from_tuples(self.df.index, names=["time_step", "time"])
        self.df.set_index(indices, inplace=True)
        self.df.to_csv(self.config.results_file)

    def random_flexibility_callback(self, inp, name):
        """
        When a flexibility offer is sent this function is called. 
        
            The offer is accepted randomly. The factor self.offer_acceptance_rate determines the
                random factor for offer acceptance. self.pos_neg_rate is the random factor for
                the direction of the flexibility. A higher rate means that more positive offers will be accepted.
            
            Constraints:
                cooldown: during $cooldown steps after a flexibility event no offer is accepted
                minimum_average_flex: min amount of flexibility to be accepted, to account for the model error
        """

        offer = inp.value
        # check if there is a flexibility provision and the cooldown is finished
        if not self.get("in_provision").value and self.cooldown_ticker == 0:
            if self.random_generator.random() < self.config.market_specs.options.offer_acceptance_rate:
                profile = None
                # if random value is below pos_neg_rate, positive offer is accepted.
                # Otherwise, negative offer
                if self.random_generator.random() < self.config.market_specs.options.pos_neg_rate:
                    if np.average(offer.pos_diff_profile) > self.config.market_specs.minimum_average_flex:
                        profile = offer.base_power_profile - offer.pos_diff_profile
                        offer.status = OfferStatus.accepted_positive

                elif np.average(offer.neg_diff_profile) > self.config.market_specs.minimum_average_flex:
                    profile = offer.base_power_profile + offer.neg_diff_profile
                    offer.status = OfferStatus.accepted_negative

                if profile is not None:
                    profile = profile.dropna()
                    profile.index += self.env.time
                    self.set("_P_external", profile)
                    self.end = profile.index[-1]
                    self.set("in_provision", True)
                    self.cooldown_ticker = self.config.market_specs.cooldown

        elif self.cooldown_ticker > 0:
            self.cooldown_ticker -= 1

        self.write_results(offer)

    def single_flexibility_callback(self, inp, name):
        """Callback to activate a single, predefined flexibility offer.

        """
        offer = inp.value
        profile = None
        if self.env.now >= self.env.config.offset + self.config.market_specs.options.start_time and not self.get("in_provision").value:
            if self.config.market_specs.options.direction == "positive":
                if np.average(offer.pos_diff_profile) > self.config.market_specs.minimum_average_flex:
                    profile = offer.base_power_profile - offer.pos_diff_profile
                    offer.status = OfferStatus.accepted_positive

            elif np.average(offer.neg_diff_profile) > self.config.market_specs.minimum_average_flex:
                profile = offer.base_power_profile + offer.neg_diff_profile
                offer.status = OfferStatus.accepted_negative

            if profile is not None:
                profile = profile.dropna()
                profile.index += self.env.time
                self.set("_P_external", profile)
                # only activate a single offer, therefore setting end to inf
                self.end = np.inf
                self.set("in_provision", True)

        self.write_results(offer)

    def custom_flexibility_callback(self, inp, name):
        """Placeholder for a custom flexibility callback"""
        pass

    def dummy_callback(self, inp, name):
        """Dummy function, that is included, when market type is not specified"""
        self.logger.warning("No market type provided. No market interaction.")

    def cleanup_results(self):
        results_file = self.config.results_file
        if not results_file:
            return
        os.remove(results_file)

    def process(self):
        while True:
            # End the provision at the appropriate time
            if self.end < self.env.time:
                self.set("in_provision", False)
            yield self.env.timeout(self.env.config.t_sample)
