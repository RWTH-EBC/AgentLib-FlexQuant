import pydantic
from pydantic import ConfigDict
from pathlib import Path
from typing import Union, List, Dict, Optional
from typing_extensions import TypedDict
from enum import Enum
from agentlib_mpc.data_structures.mpc_datamodels import MPCVariable
from agentlib.core.agent import AgentConfig
from agentlib.core.module import BaseModuleConfig
from flexibility_quantification.data_structures.mpcs import BaseMPCData, PFMPCData, NFMPCData, \
    BaselineMPCData
from flexibility_quantification.modules.flexibility_indicator import FlexibilityIndicatorModuleConfig
from flexibility_quantification.modules.flexibility_market import FlexibilityMarketModuleConfig


class ForcedOffers(Enum):
    positive = "positive"
    negative = "negative"


class ShadowMPCConfigGeneratorConfig(pydantic.BaseModel):
    """Class defining the options to of the baseline config.

    """
    weights: List[MPCVariable] = pydantic.Field(
        default=[{"name": "s_P", "value": 10}],
        description="Name and value of weights",
    )
    profile_deviation_weight: float = pydantic.Field(
        default=0,
        description="Weight of soft constraint for deviation from accepted flexible profile",
    )
    pos_flex: PFMPCData
    neg_flex: NFMPCData


class FlexibilityMarketConfig(pydantic.BaseModel):
    """Class defining the options to initialize the market.

    """
    agent_config: Union[AgentConfig, Path]
    name_of_created_file: str = "flexibility_market.json"


class FlexibilityIndicatorConfig(pydantic.BaseModel):
    """Class defining the options for the flexibility indicators.

    """
    agent_config: Union[AgentConfig, Path]
    collocation_order: int = pydantic.Field(
        default=0,
        ge=0,
        description="Order of collocation used",
    )  # TODO: get collocation order from either Baseline config or the communicated trajectories
    name_of_created_file: str = pydantic.Field(
        default="indicator.json",
        description="Name of the config that is created by the generator",
    )


class FlexQuantConfig(pydantic.BaseModel):
    """Class defining the options to initialize the FlexQuant generation.

    """

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
    indicator_config: Union[Path, FlexibilityIndicatorConfig] = pydantic.Field(
        default=None,
        description="Path to the file or dict of flexibility indicator config",
    )
    market_config: Optional[Union[Path, FlexibilityMarketConfig]] = pydantic.Field(
        default=None,
        description="Path to the file or dict of market config",
    )
    baseline_config_generator_data: Union[Path, BaselineMPCData] = pydantic.Field(
        default=None,
        description="Baseline generator data config file or dict",
    )
    shadow_mpc_config_generator_data: Union[Path, ShadowMPCConfigGeneratorConfig] = pydantic.Field(
        default=None,
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

