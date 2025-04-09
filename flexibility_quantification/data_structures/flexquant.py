import pydantic
from pydantic import ConfigDict
from pathlib import Path
from typing import Union, List, Optional
from enum import Enum
from agentlib_mpc.data_structures.mpc_datamodels import MPCVariable
from agentlib.core.agent import AgentConfig
from flexibility_quantification.data_structures.mpcs import (
    PFMPCData,
    NFMPCData,
    BaselineMPCData,
)


class ForcedOffers(Enum):
    positive = "positive"
    negative = "negative"


class ShadowMPCConfigGeneratorConfig(pydantic.BaseModel):
    """Class defining the options to of the baseline config."""

    weights: List[MPCVariable] = pydantic.Field(
        default=[],
        description="Name and value of weights",
    )
    pos_flex: PFMPCData
    neg_flex: NFMPCData

    model_config = ConfigDict(json_encoders={MPCVariable: lambda v: v.dict()})

    def __init__(self, **data):
        # Let Pydantic do its normal initialization first
        super().__init__(**data)
        # Automatically call update_weights after initialization
        self.update_weights()

    def update_weights(self):
        if self.weights:
            self.pos_flex.weights = self.weights
            self.neg_flex.weights = self.weights


class FlexibilityMarketConfig(pydantic.BaseModel):
    """Class defining the options to initialize the market."""

    agent_config: AgentConfig
    name_of_created_file: str = pydantic.Field(
        default="flexibility_market.json",
        description="Name of the config that is created by the generator",
    )


class FlexibilityIndicatorConfig(pydantic.BaseModel):
    """Class defining the options for the flexibility indicators."""

    model_config = ConfigDict(
        json_encoders={Path: str, AgentConfig: lambda v: v.model_dump()}
    )
    agent_config: AgentConfig
    name_of_created_file: str = pydantic.Field(
        default="indicator.json",
        description="Name of the config that is created by the generator",
    )


class FlexQuantConfig(pydantic.BaseModel):
    """Class defining the options to initialize the FlexQuant generation."""

    prep_time: int = pydantic.Field(
        default=1800,
        ge=0,
        unit="s",
        description="Preparation time before the flexibility event",
    )
    flex_event_duration: int = pydantic.Field(
        default=7200,
        ge=0,
        unit="s",
        description="Flexibility event duration",
    )
    market_time: int = pydantic.Field(
        default=900,
        ge=0,
        unit="s",
        description="Time for market interaction",
    )
    indicator_config: FlexibilityIndicatorConfig = pydantic.Field(
        default=None,
        description="Path to the file or dict of flexibility indicator config",
    )
    market_config: Optional[Union[FlexibilityMarketConfig, Path]] = pydantic.Field(
        default=None,
        description="Path to the file or dict of market config",
    )
    baseline_config_generator_data: BaselineMPCData = pydantic.Field(
        default=None,
        description="Baseline generator data config file or dict",
    )
    shadow_mpc_config_generator_data: ShadowMPCConfigGeneratorConfig = pydantic.Field(
        default=None,
        description="Shadow mpc generator data config file or dict",
    )
    use_CasadiSimulator: Union[bool, float] = pydantic.Field(
        default=False,
        description="If the electrical power output of the mpcs should be calculated with a defined resolution",
    )
    path_to_flex_files: Path = pydantic.Field(
        default="created_files",
        description="Path where generated files should be stored",
    )
    delete_files: bool = pydantic.Field(
        default=True,
        description="If generated files should be deleted afterwards",
    )
    overwrite_files: bool = pydantic.Field(
        default=False,
        description="If generated files should be overwritten by new files",
    )

    class Config:
        json_encoders = {Path: str}
