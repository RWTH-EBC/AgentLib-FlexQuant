import agentlib
from typing import List, Optional, Union
import pandas as pd
import numpy as np
import os
from flexibility_quantification.data_structures.flex_offer import OfferStatus, FlexOffer
import pydantic
from pathlib import Path
import matplotlib.pyplot as plt

from flexibility_quantification.data_structures.market import MarketSpecifications


bWriteResults = True


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
    # TODO: use these two
    save_results: Optional[bool] = pydantic.Field(validate_default=True, default=None)
    overwrite_result_file: Optional[bool] = pydantic.Field(default=False, validate_default=True)

    shared_variable_fields: List[str] = ["outputs"]


class FlexibilityMarketModule(agentlib.BaseModule):
    """Class to emulate flexibility market. Receives flex offers and accepts these.

    """
    config: FlexibilityMarketModuleConfig

    # TODO: cleanup
    df: pd.DataFrame = None
    df_flex_envelop: pd.DataFrame = None

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
        self.df_flex_envelop = None
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

        # write flex_envelop offers to csv file
        if self.df_flex_envelop is None:
            self.df_flex_envelop = pd.DataFrame()
        df_flex_envelop = offer.flex_envelope.as_dataframe()
        self.df_flex_envelop = pd.concat([self.df_flex_envelop, df_flex_envelop])
        self.df_flex_envelop.to_csv(self.config.market_specs.options.model_extra["results_file_offer"])

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

    def custom_flexibility_callback(self, inp, name) -> None:
        """Placeholder for a custom flexibility callback"""

        offer = inp.value
        profile: Union[None, pd.Series] = None
        time_steps: Union[None, list] = None

        #if self.env.now == self.env.config.offset + self.config.market_specs.options.start_time + 86400 and not self.get("in_provision").value:
        if self.env.now >= self.env.config.offset + self.config.market_specs.options.start_time and not self.get("in_provision").value:
        #if self.env.now == 372600 and not self.get("in_provision").value:

            bDebug: bool = True

            def draw_flex_envelope(market_type: str, offer_data: FlexOffer, event_time: int, profile_accepted: pd.Series) -> None:
                # create the folder to store the figure
                Path("plots").mkdir(parents=True, exist_ok=True)
                Path(f"plots/plots_{market_type}").mkdir(parents=True, exist_ok=True)
                flex_envelope = offer_data.flex_envelope

                # create prediction of power flexibility diagram
                time_steps_data_plot = np.delete(flex_envelope.time_steps, 0)

                fig, ax = plt.subplots()
                ax.step(time_steps_data_plot, flex_envelope.powerflex_base.values, label='powerflex_base')
                ax.step(time_steps_data_plot, flex_envelope.powerflex_pos.values, label='powerflex_pos')
                ax.step(time_steps_data_plot, flex_envelope.powerflex_neg.values, label='powerflex_neg')
                ax.set(xlabel='Zeit in s',
                       ylabel='Prädiktive Flexible Leistung $P_{el, pred}$ in kW',
                       title='Prädiktions Werte',
                       )
                ax.legend()
                plt.savefig(f"plots/plots_{market_type}/flex_offer_{event_time}_predictions.svg", format='svg')
                plt.close()

                # create Flexibility Envelope diagram
                time_steps_data_plot = flex_envelope.time_steps

                fig, ax = plt.subplots()
                ax.plot(time_steps_data_plot, flex_envelope.energyflex_base, label='energyflex_base')
                ax.plot(time_steps_data_plot, flex_envelope.energyflex_pos, label='energyflex_pos')
                ax.plot(time_steps_data_plot, flex_envelope.energyflex_neg, label='energyflex_neg')
                ax.set(xlabel='Zeit in s',
                       ylabel='Kumulierte Flexibilitäts-Energie $E_{el}$ in kWh',
                       title='Prädiktions Werte',
                       )
                ax.legend()
                plt.savefig(f"plots/plots_{market_type}/flex_offer_{event_time}_flexEnvelope.svg", format='svg')
                plt.close()

                # create selected values from Flexibility envelope diagram
                fig, ax = plt.subplots()
                ax.plot(time_steps_data_plot, flex_envelope.energyflex_base, label='energyflex_base')
                ax.plot(time_steps_data_plot, flex_envelope.energyflex_pos, label='energyflex_pos')
                ax.plot(time_steps_data_plot, flex_envelope.energyflex_neg, label='energyflex_neg')
                ax.plot(time_steps_data_plot, profile_energy.to_list(), label='ausgewählt')
                ax.set(xlabel='Zeit in s',
                       ylabel='Kumulierte Flexibilitäts-Energie $E_{el}$ in kWh',
                       title='Prädiktions Werte',
                       )
                ax.legend()
                plt.savefig(f"plots/plots_{market_type}/flex_offer_{event_time}_selectedOffer.svg", format='svg')
                plt.close()

                # create accepted offer profile diagram (which will be sent back to the building)
                base_profile = offer_data.base_power_profile
                time_steps_data_plot = np.delete(flex_envelope.time_steps, 0)

                fig, ax = plt.subplots()
                ax.step(time_steps_data_plot, profile_accepted.to_list(), label='ausgewählt')
                ax.step(time_steps_data_plot, base_profile.to_list(), label='basis')
                ax.set(xlabel='Zeit in s',
                       ylabel='Leistungsprofil $P_{el}$ in kW',
                       title='Prädiktions Werte',
                       )
                ax.legend()
                plt.savefig(f"plots/plots_{market_type}/flex_offer_{event_time}_selectedOfferProfile.svg", format='svg')
                plt.close()

                # End of draw_flex_envelope

            match self.config.market_specs.options.event_type:
                case "neg":
                    profile_energy, time_steps = flex_profile_neg(offer)

                    profile: pd.Series = convert_profile(profile_energy=profile_energy, time_steps=time_steps)

                    if bDebug:
                        draw_flex_envelope(market_type="neg", offer_data=offer, event_time=self.env.now, profile_accepted=profile)

                case "pos":
                    profile_energy, time_steps = flex_profile_pos(offer)

                    profile: pd.Series = convert_profile(profile_energy=profile_energy, time_steps=time_steps)

                    if bDebug:
                        draw_flex_envelope(market_type="pos", offer_data=offer, event_time=self.env.now, profile_accepted=profile)

                case "average":

                    # TODO: The creation of plots doesn't work anymore because one variable doesn't exist anymore
                    # find a fix !

                    profile_energy, time_steps = flex_profile_average(offer=offer)
                    profile: pd.Series = convert_profile(profile_energy=profile_energy, time_steps=time_steps)

                    #if bDebug:
                    #    draw_flex_envelope(market_type="average", offer_data=offer, event_time=self.env.now, profile_accepted=profile)

                case "real":
                    profile_energy, time_steps = flex_profile_real(offer=offer)
                    profile: pd.Series = convert_profile(profile_energy=profile_energy, time_steps=time_steps)

                    if bDebug:
                        draw_flex_envelope(market_type="real", offer_data=offer, event_time=self.env.now, profile_accepted=profile)

                case _:
                    print("Wrong Market Event Type selected ! \nReal Profile was automatically selected.")

                    profile_energy, time_steps = flex_profile_real(offer=offer)
                    profile: pd.Series = convert_profile(profile_energy=profile_energy, time_steps=time_steps)

                    if bDebug:
                        draw_flex_envelope(market_type="real", offer_data=offer, event_time=self.env.now, profile_accepted=profile)

        if profile is not None:
            offer.status = OfferStatus.accepted

            # time_step = profile.index[-1] - profile.index[-2]
            # temp_Series = pd.Series(data=[0], index=[(profile.index[-1] + time_step)])
            # profile = pd.concat([profile, temp_Series])

            profile = profile.ffill()
            profile.index += self.env.time
            self.set("_P_external", profile)

            if self.config.market_specs.options.multi_offer:
                self.end = self.env.now + time_steps[-1]
            else:
                self.end = np.inf

            self.set("in_provision", True)

        if bWriteResults:
            self.write_results(offer)

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


def flex_profile_neg(offer: FlexOffer) -> tuple[pd.Series, list]:
    return offer.flex_envelope.energyflex_neg, offer.flex_envelope.time_steps


def flex_profile_pos(offer: FlexOffer) -> tuple[pd.Series, list]:
    return offer.flex_envelope.energyflex_pos, offer.flex_envelope.time_steps


def flex_profile_average(offer: FlexOffer) -> tuple[pd.Series, list]:
    profile_energy_neg, time_steps = flex_profile_neg(offer)
    profile_power_neg: pd.Series = convert_profile(profile_energy=profile_energy_neg,
                                                   time_steps=time_steps)

    profile_energy_pos, time_steps = flex_profile_pos(offer)
    profile_power_pos: pd.Series = convert_profile(profile_energy=profile_energy_pos,
                                                   time_steps=time_steps)

    profile = (profile_power_neg + profile_power_pos) / 2

    return profile, time_steps


# This Global parameter is needed for flex_profile_real
randGenSeedCount = 0


def flex_profile_real(offer: FlexOffer) -> tuple[pd.Series, list]:
    # Todo: write a real profile function
    # create random generator
    global randGenSeedCount
    if randGenSeedCount is None:
        randGenSeedCount = 0
    else:
        randGenSeedCount += 1

    random_generator = np.random.default_rng(randGenSeedCount)

    profile_energy_neg, time_steps = flex_profile_neg(offer)
    profile_energy_pos, _ = flex_profile_pos(offer)
    p_el_min: float = offer.flex_envelope.p_el_min
    p_el_max: float = offer.flex_envelope.p_el_max

    time_step = time_steps[1] - time_steps[0]
    profile_selected: list[float] = [0.0]

    e_el_max: float = p_el_max * (time_step/3600)
    e_el_min: float = p_el_min * (time_step/3600)

    for iIdx in range(1, profile_energy_neg.size):

        # TODO: Find a better solution!

        #usable_max_value: float = min(e_el_max + profile_selected[iIdx-1], profile_energy_neg[iIdx])
        #usable_min_value: float = max(e_el_min + profile_selected[iIdx-1], profile_energy_pos[iIdx])

        # generate a random number between -usable_max_value and +usable_max_value
        #gen_number = random_generator.uniform(usable_min_value, usable_max_value)
        gen_number = random_generator.uniform(profile_selected[iIdx-1], profile_energy_neg[iIdx])
        profile_selected.append(gen_number)

    profile: pd.Series = pd.Series(profile_selected)

    return profile, time_steps


def convert_profile(profile_energy: pd.Series, time_steps: list) -> pd.Series:
    """
    convert_profile: takes the accumulated energy profile from the selected offer,
                     computes the slope of the offer to get the kW profile, for passing it to the building.
    """
    flex_power = []
    flex_power_levels = []

    # reset index of profile_energy for easier looping
    profile_energy = profile_energy.reset_index(drop=True)
    # time_steps = time_steps.reset_index(drop=True)

    # convert the (k)Wh to (k)W before creating a series
    time_step = 0
    for iIdx in range(len(profile_energy) - 1):
        time_step = (time_steps[iIdx + 1] - time_steps[iIdx])
        flex_power.append((profile_energy[iIdx + 1] - profile_energy[iIdx]) / (time_step / 3600))

    return pd.Series(data=flex_power, index=np.delete(time_steps, 0))

