from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import pydantic
from pydantic import ConfigDict, model_validator

from agentlib.core.agent import AgentConfig
from agentlib_mpc.data_structures.mpc_datamodels import MPCVariable
from flexibility_quantification.data_structures.mpcs import (
    BaselineMPCData,
    NFMPCData,
    PFMPCData,
)


class ForcedOffers(Enum):
    positive = "positive"
    negative = "negative"


class ShadowMPCConfigGeneratorConfig(pydantic.BaseModel):
    """Class defining the options to initialize the shadow mpc config generation."""
    model_config = ConfigDict(
        json_encoders={MPCVariable: lambda v: v.dict()},
        extra='forbid'
    )    
    weights: List[MPCVariable] = pydantic.Field(
        default=[],
        description="Name and value of weights",
    )
    pos_flex: PFMPCData = pydantic.Field(
        default=None,
        description="Data for PF-MPC"
    )
    neg_flex: NFMPCData = pydantic.Field(
        default=None,
        description="Data for NF-MPC"
    )
    @model_validator(mode="after")
    def assign_weights_to_flex(self):
        if self.pos_flex is None:
            raise ValueError("Missing required field: 'pos_flex' specifying the pos flex cost function.")
        if self.neg_flex is None:
            raise ValueError("Missing required field: 'neg_flex' specifying the neg flex cost function.")
        if self.weights:
            self.pos_flex.weights = self.weights
            self.neg_flex.weights = self.weights
        return self


class FlexibilityMarketConfig(pydantic.BaseModel):
    """Class defining the options to initialize the market."""
    model_config = ConfigDict(
        extra='forbid'
    )
    agent_config: AgentConfig
    name_of_created_file: str = pydantic.Field(
        default="flexibility_market.json",
        description="Name of the config that is created by the generator",
    )


class FlexibilityIndicatorConfig(pydantic.BaseModel):
    """Class defining the options for the flexibility indicators."""
    model_config = ConfigDict(
        json_encoders={Path: str, AgentConfig: lambda v: v.model_dump()},
        extra='forbid'
    )
    agent_config: AgentConfig
    name_of_created_file: str = pydantic.Field(
        default="indicator.json",
        description="Name of the config that is created by the generator",
    )


class FlexQuantConfig(pydantic.BaseModel):
    """Class defining the options to initialize the FlexQuant generation."""
    model_config = ConfigDict(
        json_encoders={Path: str},
        extra='forbid'
    )
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
    indicator_config: Union[FlexibilityIndicatorConfig, Path] = pydantic.Field(
        description="Path to the file or dict of flexibility indicator config",
    )
    market_config: Optional[Union[FlexibilityMarketConfig, Path]] = pydantic.Field(
        default=None,
        description="Path to the file or dict of market config",
    )
    baseline_config_generator_data: Union[BaselineMPCData, Path] = pydantic.Field(
        description="Baseline generator data config file or dict",
    )
    shadow_mpc_config_generator_data: Union[ShadowMPCConfigGeneratorConfig, Path] = pydantic.Field(
        description="Shadow mpc generator data config file or dict",
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