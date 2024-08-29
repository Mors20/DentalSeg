

import dynamic_network_architectures
from copy import deepcopy
from functools import lru_cache, partial
from typing import Union, Tuple, List, Type, Callable

import numpy as np
import torch
from utilities.label_handler import LabelManager
from torch import nn

from batchgenerators.utilities.file_and_folder_operations import load_json, join





class ConfigurationManager(object):
    def __init__(self, configuration_dict: dict):
        self.configuration = configuration_dict


    def __repr__(self):
        return self.configuration.__repr__()

    @property
    def data_identifier(self) -> str:
        return self.configuration['data_identifier']

    @property
    def preprocessor_name(self) -> str:
        return self.configuration['preprocessor_name']

    @property
    def batch_size(self) -> int:
        return self.configuration['batch_size']

    @property
    def patch_size(self) -> List[int]:
        return self.configuration['patch_size']

    @property
    def median_image_size_in_voxels(self) -> List[int]:
        return self.configuration['median_image_size_in_voxels']

    @property
    def spacing(self) -> List[float]:
        return self.configuration['spacing']

    @property
    def normalization_schemes(self) -> List[str]:
        return self.configuration['normalization_schemes']

    @property
    def use_mask_for_norm(self) -> List[bool]:
        return self.configuration['use_mask_for_norm']

    @property
    def network_arch_class_name(self) -> str:
        return self.configuration['architecture']['network_class_name']

    @property
    def network_arch_init_kwargs(self) -> dict:
        return self.configuration['architecture']['arch_kwargs']
    @property
    def network_arch_init_kwargs_req_import(self) -> Union[Tuple[str, ...], List[str]]:
        return self.configuration['architecture']['_kw_requires_import']

    @property
    def pool_op_kernel_sizes(self) -> Tuple[Tuple[int, ...], ...]:
        return self.configuration['architecture']['arch_kwargs']['strides']


    @property
    def resampling_fn_data_kwargs(self):
        return self.configuration['resampling_fn_data_kwargs']

    @property
    def resampling_fn_seg_kwargs(self):
        return self.configuration['resampling_fn_seg_kwargs']
    
    @property
    def resampling_fn_probabilities_kwargs(self):
        return self.configuration['resampling_fn_probabilities_kwargs']


    @property
    def batch_dice(self) -> bool:
        return self.configuration['batch_dice']

    @property
    def next_stage_names(self) -> Union[List[str], None]:
        ret = self.configuration.get('next_stage')
        if ret is not None:
            if isinstance(ret, str):
                ret = [ret]
        return ret

    @property
    def previous_stage_name(self) -> Union[str, None]:
        return self.configuration.get('previous_stage')


class PlansManager(object):
    def __init__(self, plans_file_or_dict: Union[str, dict]):
        """
        Why do we need this?
        1) resolve inheritance in configurations
        2) expose otherwise annoying stuff like getting the label manager or IO class from a string
        3) clearly expose the things that are in the plans instead of hiding them in a dict
        4) cache shit

        This class does not prevent you from going wild. You can still use the plans directly if you prefer
        (PlansHandler.plans['key'])
        """
        self.plans = plans_file_or_dict if isinstance(plans_file_or_dict, dict) else load_json(plans_file_or_dict)

    def __repr__(self):
        return self.plans.__repr__()

    def _internal_resolve_configuration_inheritance(self, configuration_name: str,
                                                    visited: Tuple[str, ...] = None) -> dict:
        if configuration_name not in self.plans['configurations'].keys():
            raise ValueError(f'The configuration {configuration_name} does not exist in the plans I have. Valid '
                             f'configuration names are {list(self.plans["configurations"].keys())}.')
        configuration = deepcopy(self.plans['configurations'][configuration_name])
        if 'inherits_from' in configuration:
            parent_config_name = configuration['inherits_from']

            if visited is None:
                visited = (configuration_name,)
            else:
                if parent_config_name in visited:
                    raise RuntimeError(f"Circular dependency detected. The following configurations were visited "
                                       f"while solving inheritance (in that order!): {visited}. "
                                       f"Current configuration: {configuration_name}. Its parent configuration "
                                       f"is {parent_config_name}.")
                visited = (*visited, configuration_name)

            base_config = self._internal_resolve_configuration_inheritance(parent_config_name, visited)
            base_config.update(configuration)
            configuration = base_config
        return configuration

    @lru_cache(maxsize=10)
    def get_configuration(self, configuration_name: str):
        if configuration_name not in self.plans['configurations'].keys():
            raise RuntimeError(f"Requested configuration {configuration_name} not found in plans. "
                               f"Available configurations: {list(self.plans['configurations'].keys())}")

        configuration_dict = self._internal_resolve_configuration_inheritance(configuration_name)
        return ConfigurationManager(configuration_dict)

    @property
    def dataset_name(self) -> str:
        return self.plans['dataset_name']

    @property
    def plans_name(self) -> str:
        return self.plans['plans_name']

    @property
    def original_median_spacing_after_transp(self) -> List[float]:
        return self.plans['original_median_spacing_after_transp']

    @property
    def original_median_shape_after_transp(self) -> List[float]:
        return self.plans['original_median_shape_after_transp']

    # @property
    # @lru_cache(maxsize=1)
    # def image_reader_writer_class(self) -> Type[BaseReaderWriter]:
    #     return recursive_find_reader_writer_by_name(self.plans['image_reader_writer'])

    @property
    def transpose_forward(self) -> List[int]:
        return self.plans['transpose_forward']

    @property
    def transpose_backward(self) -> List[int]:
        return self.plans['transpose_backward']

    @property
    def available_configurations(self) -> List[str]:
        return list(self.plans['configurations'].keys())


        return experiment_planner

    @property
    def experiment_planner_name(self) -> str:
        return self.plans['experiment_planner_used']



    def get_label_manager(self, dataset_json: dict, **kwargs) -> LabelManager:
        return LabelManager(label_dict=dataset_json['labels'],
                                        regions_class_order=dataset_json.get('regions_class_order'),
                                        **kwargs)

    @property
    def foreground_intensity_properties_per_channel(self) -> dict:
        if 'foreground_intensity_properties_per_channel' not in self.plans.keys():
            if 'foreground_intensity_properties_by_modality' in self.plans.keys():
                return self.plans['foreground_intensity_properties_by_modality']
        return self.plans['foreground_intensity_properties_per_channel']



