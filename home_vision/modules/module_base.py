"""Base class for HomeVision modules"""
from __future__ import annotations
import logging
import time

from abc import abstractmethod
from typing import Any, Generic, Type, TypeVar
from pydantic import BaseModel #pylint: disable=no-name-in-module

from home_vision.common.configurable import Configurable
from home_vision.common.registrable import Registrable

class ModuleInput(BaseModel):
    """Base Module Input Types"""
    class Config:
        """Pydantic model config"""
        frozen=True
        extra='forbid'
        arbitrary_types_allowed = True

class ModuleOutput(BaseModel):
    """Base Output Types"""
    class Config:
        """Pydantic model config"""
        frozen=True
        extra='forbid'
        arbitrary_types_allowed = True

InputT = TypeVar('InputT', bound='ModuleInput')
OutputT = TypeVar('OutputT', bound='ModuleOutput')
ConfigT = TypeVar('ConfigT', bound='ModuleConfig')

class Module(Configurable, Generic[InputT, OutputT, ConfigT]):
    """Base class for HomeVision modules"""
    input_types: Type[InputT] = InputT
    output_types: Type[OutputT] = OutputT

    @property
    @abstractmethod
    def module_name(self) -> str:
        """Module's name"""
        raise NotImplementedError

    @property
    @abstractmethod
    def config_type(self) -> BaseModel:
        """Module's config type"""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_config(cls, config: ConfigT) -> Module:
        pass

    def process(self, inputs: InputT) -> OutputT:
        """HomeVision module's process function"""
        time_s = time.perf_counter()
        assert isinstance(inputs, self.input_types), \
            f"{self.module_name}'s input is type {type(input)}; \
            doesn't match {self.input_types}"
        outputs = self._process(inputs=inputs)
        assert isinstance(outputs, self.output_types), \
            f"{self.module_name}'s outputs is type {type(outputs)}; \
            doesn't match {self.output_types}"
        time_e = time.perf_counter()
        logging.debug("%s: %.2f ms", self.module_name, (time_e - time_s) * 1000)
        return outputs

    @abstractmethod
    def _process(self, inputs: InputT) -> OutputT:
        """HomeVision module's process function"""
        pass

class BaseConfig(Registrable, BaseModel, Generic[ConfigT]):
    """Base HomeVision config"""
    class Config:
        """Pydantic model config"""
        frozen=True
        extra='forbid'


class ModuleConfig(BaseConfig, Generic[ConfigT]):
    """Base Module config class"""
    def __init__(self, **data: Any):
        if 'method' not in data:
            raise KeyError("Please define method for the module!")
        if 'config' not in data:
            raise KeyError("Please define config for the module!")
        data["config"] = self.by_name(data["method"])(**data['config'])
        super().__init__(**data)
