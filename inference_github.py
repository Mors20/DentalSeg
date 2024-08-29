import inspect
import itertools
import multiprocessing
import os
from glob import glob
from copy import deepcopy
from time import sleep
from typing import Tuple, Union, List, Optional
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
import numpy as np
import torch
from scipy.ndimage import binary_erosion
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json
from torch import nn
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from utilities.plans_handler import PlansManager, ConfigurationManager
from utilities.get_network_from_plans import get_network_from_plans
from utilities.simpleitk_reader_writer import SimpleITKIO
from utilities.croppoing import crop_to_nonzero
from utilities.resampling import compute_new_shape, resample_data_or_seg_to_shape
from utilities.normalization import ZScoreNormalization, CTNormalization
from utilities.sliding_window_prediction import compute_gaussian , compute_steps_for_sliding_window
from skimage import morphology
from scipy import ndimage
from skimage.segmentation import watershed
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt,label, find_objects
from skimage.morphology import remove_small_objects
import json
class nnUNetPredictor(object):
    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_device: bool = True,
                 device: torch.device = torch.device('cuda'),
                 allow_tqdm: bool = True):

        self.allow_tqdm = allow_tqdm

        self.plans_manager, self.configuration_manager, self.list_of_parameters, self.network, self.dataset_json, \
        self.trainer_name, self.allowed_mirroring_axes, self.label_manager = None, None, None, None, None, None, None, None
        
        self.tile_step_size = tile_step_size
        self.use_gaussian = use_gaussian
        self.use_mirroring = use_mirroring
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        else:
            print(f'perform_everything_on_device=True is only supported for cuda devices! Setting this to False')
            perform_everything_on_device = False
        self.device = device
        self.perform_everything_on_device = perform_everything_on_device

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        """
        This is used when making predictions with a trained model
        """
        if use_folds is None:
            use_folds = nnUNetPredictor.auto_detect_available_folds(model_training_output_dir, checkpoint_name)

        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)
        if use_folds == 'all':
            use_folds = [0,1,2,3,4]
        if isinstance(use_folds, str):
            use_folds = [use_folds]
        
        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != 'all' else f
            checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                    map_location=torch.device('cpu'))
            if i == 0:
                trainer_name = checkpoint['trainer_name']
                configuration_name = checkpoint['init_args']['configuration']
                inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                    'inference_allowed_mirroring_axes' in checkpoint.keys() else None

            parameters.append(checkpoint['network_weights'])

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = 1

        network = get_network_from_plans(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            deep_supervision=False
        )

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)

    def manual_initialization(self, network: nn.Module, plans_manager: PlansManager,
                              configuration_manager: ConfigurationManager, parameters: Optional[List[dict]],
                              dataset_json: dict, trainer_name: str,
                              inference_allowed_mirroring_axes: Optional[Tuple[int, ...]]):
        """
        This is used by the nnUNetTrainer to initialize nnUNetPredictor for the final validation
        """
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        allow_compile = True
        allow_compile = allow_compile and ('nnUNet_compile' in os.environ.keys()) and (
                    os.environ['nnUNet_compile'].lower() in ('true', '1', 't'))
        allow_compile = allow_compile and not isinstance(self.network, OptimizedModule)
        if isinstance(self.network, DistributedDataParallel):
            allow_compile = allow_compile and isinstance(self.network.module, OptimizedModule)
        if allow_compile:
            print('Using torch.compile')
            self.network = torch.compile(self.network)

    @staticmethod
    def auto_detect_available_folds(model_training_output_dir, checkpoint_name):
        print('use_folds is None, attempting to auto detect available folds')
        fold_folders = subdirs(model_training_output_dir, prefix='fold_', join=False)
        fold_folders = [i for i in fold_folders if i != 'fold_all']
        fold_folders = [i for i in fold_folders if isfile(join(model_training_output_dir, i, checkpoint_name))]
        use_folds = [int(i.split('_')[-1]) for i in fold_folders]
        print(f'found the following folds: {use_folds}')
        return use_folds


    def _normalize(self, data: np.ndarray, seg: np.ndarray, configuration_manager: ConfigurationManager,
                   foreground_intensity_properties_per_channel: dict) -> np.ndarray:
        for c in range(data.shape[0]):
            scheme = configuration_manager.normalization_schemes[c]
            if scheme  == 'ZScoreNormalization':
                normalizer = ZScoreNormalization(use_mask_for_norm=configuration_manager.use_mask_for_norm[c],
                                          intensityproperties=foreground_intensity_properties_per_channel[str(c)])
            elif scheme == 'CTNormalization':
                normalizer = CTNormalization(use_mask_for_norm=configuration_manager.use_mask_for_norm[c],
                                          intensityproperties=foreground_intensity_properties_per_channel[str(c)])
            else:
                raise NotImplementedError 
            data[c] = normalizer.run(data[c], seg[0])               
        return data 

    def preprocess(self, data, data_properties):
        data = data.transpose([0, *[i + 1 for i in self.plans_manager.transpose_forward]])
        original_spacing = [data_properties['spacing'][i] for i in self.plans_manager.transpose_forward]
        shape_before_cropping = data.shape[1:]
        data_properties['shape_before_cropping'] = shape_before_cropping
        data, seg, bbox = crop_to_nonzero(data)
        data_properties['bbox_used_for_cropping'] = bbox
        data_properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]
        target_spacing = self.configuration_manager.spacing 
        if len(target_spacing) < len(data.shape[1:]):
            # target spacing for 2d has 2 entries but the data and original_spacing have three because everything is 3d
            # in 2d configuration we do not change the spacing between slices
            target_spacing = [original_spacing[0]] + target_spacing
        new_shape = compute_new_shape(data.shape[1:] , original_spacing, target_spacing)
        data = self._normalize(data, seg, self.configuration_manager,
                               self.plans_manager.foreground_intensity_properties_per_channel)
        data = resample_data_or_seg_to_shape(data,new_shape,original_spacing,target_spacing,**self.configuration_manager.resampling_fn_data_kwargs)
        seg = resample_data_or_seg_to_shape(seg,new_shape,original_spacing,target_spacing,**self.configuration_manager.resampling_fn_seg_kwargs)
        if np.max(seg) > 127:
            seg = seg.astype(np.int16)
        else:
            seg = seg.astype(np.int8)
        return data, seg,data_properties

    def predict_from_file(self, data, data_properties ):
            data_temp = data.copy()
            data_temp,seg, data_preproperties = self.preprocess(data_temp, data_properties)
            prediction = self.predict_logits_from_preprocessed_data(data_temp)
            segmentation_final, properties_dict = self.export_prediction_from_logits(prediction,data_preproperties,self.configuration_manager,self.plans_manager,self.dataset_json)
            return segmentation_final, properties_dict
        

    def predict_logits_from_preprocessed_data(self, data):
        data = torch.tensor(data).to(self.device)
        with torch.no_grad():
            prediction = None
            for params in self.list_of_parameters:
                if not isinstance(self.network, OptimizedModule):
                    self.network.load_state_dict(params)
                else:
                    self.network._orig_mod.load_state_dict(params)
                if prediction is None:
                    prediction = self.predict_sliding_window_return_logits(data).to('cpu')
                else:
                    prediction += self.predict_sliding_window_return_logits(data).to('cpu')
                    
            if len(self.list_of_parameters) > 1:
                prediction /= len(self.list_of_parameters)
            # prediction = prediction.to('cpu')
        return prediction


    def export_prediction_from_logits(self,predicted_logits: Union[np.ndarray, torch.Tensor], properties_dict: dict,
                                  configuration_manager: ConfigurationManager,
                                  plans_manager: PlansManager,
                                  dataset_json: dict):
        label_manager = plans_manager.get_label_manager(dataset_json)
        current_spacing = configuration_manager.spacing if \
            len(configuration_manager.spacing) == \
            len(properties_dict['shape_after_cropping_and_before_resampling']) else \
            [properties_dict['spacing'][0], *configuration_manager.spacing]
        predicted_logits = resample_data_or_seg_to_shape(predicted_logits,
                                                properties_dict['shape_after_cropping_and_before_resampling'],
                                                current_spacing,
                                                properties_dict['spacing'],**configuration_manager.resampling_fn_probabilities_kwargs)
        predicted_probabilities = label_manager.apply_inference_nonlin(predicted_logits)
        del predicted_logits
        segmentation = label_manager.convert_probabilities_to_segmentation(predicted_probabilities)
        if isinstance(segmentation, torch.Tensor):
            segmentation = segmentation.cpu().numpy()

        # put segmentation in bbox (revert cropping)
        segmentation_reverted_cropping = np.zeros(properties_dict['shape_before_cropping'],
                                                dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16)
        slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
        segmentation_reverted_cropping[slicer] = segmentation
        segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(plans_manager.transpose_backward)
        del segmentation
        segmentation_final = segmentation_reverted_cropping
        return segmentation_final, properties_dict

 

    def _internal_get_sliding_window_slicers(self, image_size: Tuple[int, ...]):
        slicers = []
        if len(self.configuration_manager.patch_size) < len(image_size):
            assert len(self.configuration_manager.patch_size) == len(
                image_size) - 1, 'if tile_size has less entries than image_size, ' \
                                 'len(tile_size) ' \
                                 'must be one shorter than len(image_size) ' \
                                 '(only dimension ' \
                                 'discrepancy of 1 allowed).'
            steps = compute_steps_for_sliding_window(image_size[1:], self.configuration_manager.patch_size,
                                                     self.tile_step_size)
            for d in range(image_size[0]):
                for sx in steps[0]:
                    for sy in steps[1]:
                        slicers.append(
                            tuple([slice(None), d, *[slice(si, si + ti) for si, ti in
                                                     zip((sx, sy), self.configuration_manager.patch_size)]]))
        else:
            steps = compute_steps_for_sliding_window(image_size, self.configuration_manager.patch_size,
                                                     self.tile_step_size)

            for sx in steps[0]:
                for sy in steps[1]:
                    for sz in steps[2]:
                        slicers.append(
                            tuple([slice(None), *[slice(si, si + ti) for si, ti in
                                                  zip((sx, sy, sz), self.configuration_manager.patch_size)]]))
        return slicers

    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        prediction = self.network(x)

        # if mirror_axes is not None:
        #     # check for invalid numbers in mirror_axes
        #     # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
        #     assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'

        #     mirror_axes = [m + 2 for m in mirror_axes]
        #     axes_combinations = [
        #         c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
        #     ]
        #     for axes in axes_combinations:
        #         prediction += torch.flip(self.network(torch.flip(x, axes)), axes)
        #     prediction /= (len(axes_combinations) + 1)
        return prediction

    def _internal_predict_sliding_window_return_logits(self,
                                                       data: torch.Tensor,
                                                       slicers,
                                                       do_on_device: bool = True,
                                                       ):
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device('cpu')

        try:
            torch.cuda.empty_cache()

            # move data to device

            data = data.to(results_device)

            # preallocate arrays

            predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                           dtype=torch.half,
                                           device=results_device)
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

            if self.use_gaussian:
                gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                            value_scaling_factor=10,
                                            device=results_device)
            else:
                gaussian = 1


            for sl in tqdm(slicers, disable=not self.allow_tqdm):
                workon = data[sl][None]
                workon = workon.to(self.device)

                prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)

                if self.use_gaussian:
                    prediction *= gaussian
                predicted_logits[sl] += prediction
                n_predictions[sl[1:]] += gaussian

            predicted_logits /= n_predictions
            # check for infs
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '
                                   'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
                                   'predicted_logits to fp32')
        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            torch.cuda.empty_cache()
            raise e
        return predicted_logits

    @torch.inference_mode()
    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) \
            -> Union[np.ndarray, torch.Tensor]:
        assert isinstance(input_image, torch.Tensor)
        self.network = self.network.to(self.device)
        self.network.eval()

        torch.cuda.empty_cache()

        with torch.no_grad():
            with torch.autocast(self.device.type, enabled=True):
                assert len(input_image.shape) == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'




                data, slicer_revert_padding = pad_nd_image(input_image, self.configuration_manager.patch_size,
                                                           'constant', {'value': 0}, True,
                                                           None)

                slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

                # preallocate results and num_predictions
                results_device = self.device 

                try:
                    data = data.to(self.device)
                    predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                                   dtype=torch.half,
                                                   device=results_device)
                    n_predictions = torch.zeros(data.shape[1:], dtype=torch.half,
                                                device=results_device)
                    if self.use_gaussian:
                        gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                                    value_scaling_factor=1000,
                                                    device=results_device)
                except RuntimeError:
                    # sometimes the stuff is too large for GPUs. In that case fall back to CPU
                    results_device = torch.device('cpu')
                    data = data.to(results_device)
                    predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                                   dtype=torch.half,
                                                   device=results_device)
                    n_predictions = torch.zeros(data.shape[1:], dtype=torch.half,
                                                device=results_device)
                    if self.use_gaussian:
                        gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                                    value_scaling_factor=1000,
                                                    device=results_device)
                finally:
                    torch.cuda.empty_cache()


                for sl in tqdm(slicers, disable=not self.allow_tqdm):
                    workon = data[sl][None]
                    workon = workon.to(self.device, non_blocking=False)

                    prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)

                    predicted_logits[sl] += (prediction * gaussian if self.use_gaussian else prediction)
                    n_predictions[sl[1:]] += (gaussian if self.use_gaussian else 1)

                predicted_logits /= n_predictions
        torch.cuda.empty_cache()
        return predicted_logits[tuple([slice(None), *slicer_revert_padding[1:]])]



class segmentation_algorithm(object):
    def __init__(self):

        model_paths=('model1Path',
                         'model2Path',
                         'model3Path')
        model1_path, model2_path,model3_path = model_paths
        self.new_old_mapping = load_json(join('./utilities', 'new_old_mapping.json'))
        self.predictor1 = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            device=torch.device('cuda', 0),
            )
        self.predictor2 = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            device=torch.device('cuda', 0),
            )
        self.predictor3 = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            device=torch.device('cuda', 0),
            )
        self.predictor1.initialize_from_trained_model_folder(model1_path,
            use_folds=(4,),
            checkpoint_name='checkpoint_final.pth',
            )
        self.predictor2.initialize_from_trained_model_folder(model2_path,
            use_folds=(4,),
            checkpoint_name='checkpoint_final.pth',
            )
        self.predictor3.initialize_from_trained_model_folder(model3_path,
            use_folds='all',
            checkpoint_name='checkpoint_final.pth',
            )

    
    def get_data_properties(self,input_image:sitk.Image):
        images = []
        spacings = []
        origins = []
        directions = []

        spacings_for_nnunet = []

        spacings.append(input_image.GetSpacing())
        origins.append(input_image.GetOrigin())
        directions.append(input_image.GetDirection())
        if directions[0] != (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
            input_image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        npy_image = sitk.GetArrayFromImage(input_image)
        
        if npy_image.ndim == 2:
            # 2d
            npy_image = npy_image[None, None]
            max_spacing = max(spacings[-1])
            spacings_for_nnunet.append((max_spacing * 999, *list(spacings[-1])[::-1]))
        elif npy_image.ndim == 3:
            # 3d, as in original nnunet
            npy_image = npy_image[None]
            spacings_for_nnunet.append(list(spacings[-1])[::-1])
        elif npy_image.ndim == 4:
            # 4d, multiple modalities in one file
            spacings_for_nnunet.append(list(spacings[-1])[::-1][1:])
            pass
        else:
            raise RuntimeError(f"Unexpected number of dimensions: {npy_image.ndim} in file {f}")

        images.append(npy_image)
        spacings_for_nnunet[-1] = list(np.abs(spacings_for_nnunet[-1]))



        stacked_images = np.vstack(images)
        dict = {
            'sitk_stuff': {
                # this saves the sitk geometry information. This part is NOT used by nnU-Net!
                'spacing': spacings[0],
                'origin': origins[0],
                'direction': directions[0]
            },
            # the spacing is inverted with [::-1] because sitk returns the spacing in the wrong order lol. Image arrays
            # are returned x,y,z but spacing is returned z,y,x. Duh.
            'spacing': spacings_for_nnunet[0]
        }
        return stacked_images.astype(np.float32), dict
    

    def cal_max_component(self,mask):
        structure = np.ones((3, 3, 3), dtype=np.int16)
        labeled_array, _  = label(mask, structure)
        sizes = np.bincount(labeled_array.ravel())
        sizes[0] = 0
        max_label = sizes.argmax()
        if max_label == 0:
            return np.zeros_like(mask)
        return (labeled_array == max_label)
    

    def label_left_right(self,mask):
        dividing_line = mask.shape[-1]//2
        mask_left , mask_right= mask.copy(), mask.copy()
        mask_left[:,:,dividing_line:] , mask_right[:,:,:dividing_line] = 0 , 0
        return mask_left, mask_right
    
    def postprocess_1(self,segmentation_result):
        Lower_Jawbone = self.cal_max_component(segmentation_result == 1) 
        Upper_Jawbone = self.cal_max_component(segmentation_result == 2) 

        Canal = segmentation_result == 3 
        Canal = remove_small_objects(Canal,min_size=int(Canal.sum()*0.01),connectivity=2).astype(np.uint8)
        Canal_left, Canal_right = self.label_left_right(Canal)

        Sinus = segmentation_result == 4
        Sinus = remove_small_objects(Sinus,min_size=int(Sinus.sum()*0.01),connectivity=2).astype(np.uint8)
        Sinus_left, Sinus_right = self.label_left_right(Sinus)

        Pharynx = self.cal_max_component(segmentation_result == 5) 
        Teeth = (segmentation_result == 6)
        
        processed_result = np.zeros_like(segmentation_result)
        processed_result[Lower_Jawbone>0] = 1
        processed_result[Upper_Jawbone>0] = 2
        processed_result[Canal_left>0] = 3
        processed_result[Canal_right>0] = 4
        processed_result[Sinus_left>0] = 5
        processed_result[Sinus_right>0] = 6
        processed_result[Pharynx>0] = 7
        processed_result[Teeth>0] = 8
        jawbone = np.zeros_like(processed_result)
        jawbone[processed_result == 1] = 1
        jawbone[processed_result == 2] = 1
        non_zero_indices = np.argwhere(jawbone != 0)
        min_z, min_y, min_x = non_zero_indices.min(axis=0)
        max_z, max_y, max_x = non_zero_indices.max(axis=0)
        min_z , min_y, min_x = max(0,min_z-5),max(0,min_y-5),max(0,min_x-5)
        max_z, max_y, max_x = min(segmentation_result.shape[0]-1,max_z+5),min(segmentation_result.shape[1]-1,max_y+5),min(segmentation_result.shape[2]-1,max_x+5)
        outside_bbox = np.ones_like(segmentation_result, dtype=bool)
        outside_bbox[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1] = 0
        label_to_process = [3,4,5,6,7,8]
        for label in label_to_process:
            processed_result[outside_bbox & (processed_result == label)] = 0
        return processed_result.astype(np.uint8)

    
    def predict(self, input_image: sitk.Image):


        data, data_properties = self.get_data_properties(input_image)
        segmentation_result1, properties_dict = self.predictor1.predict_from_file(data , data_properties)
        segmentation_result1 = self.postprocess_1(segmentation_result1)

        teeth_mask = (segmentation_result1==8).astype(np.uint8)
        if teeth_mask.sum() == 0:
            true_id_teeth = np.zeros_like(segmentation_result1)
        else:
            index = np.where(teeth_mask>0)
            x_min,x_max,y_min,y_max,z_min,z_max = max(0,index[0].min()-10), min(index[0].max()+10,data.shape[1]), max(0,index[1].min()-10), \
                min(data.shape[2],index[1].max()+10), max(0,index[2].min()-10), min(data.shape[3],index[2].max()+10)

            img_to_3 = data[:,x_min:x_max, y_min:y_max,z_min:z_max]
            teeth_mask_crop = teeth_mask[:,x_min:x_max, y_min:y_max,z_min:z_max]
            segmentation_result2, properties_dict = self.predictor2.predict_from_file(teeth_mask_crop , data_properties)

            seeds_map = np.zeros_like(segmentation_result2)
            seeds_map[segmentation_result2 == 1] = 1
            seeds_map = morphology.remove_small_objects(seeds_map.astype(bool), 10, connectivity=1)
            markers = ndimage.label(seeds_map)[0]
            distance = ndimage.distance_transform_edt(teeth_mask_crop)
            sub_teeth_array = watershed(-distance, markers=markers, mask=teeth_mask_crop)
            teeth_array = np.zeros_like(segmentation_result1)
            teeth_array[x_min:x_max, y_min:y_max,z_min:z_max] =sub_teeth_array

            segmentation_result3, properties_dict = self.predictor3.predict_from_file(img_to_3 , data_properties)
            whole_id_mask = np.zeros_like(segmentation_result1)
            whole_id_mask[x_min:x_max, y_min:y_max,z_min:z_max] =segmentation_result3
            

            # voting
            labels = np.unique(teeth_array[teeth_array != 0])
            true_id_teeth = np.zeros_like(teeth_array)
            for label in labels:
                single_id = np.where(teeth_array == label, 1, 0) * whole_id_mask 
                single = single_id[single_id != 0]
                dict = {}
                ids = np.unique(single)
                if len(ids) > 1:
                    for id in ids:
                        dict[id] = np.sum(single == id)
                    true_id = max(dict, key=lambda i: dict[i])
                    if true_id != 2:
                        true_id_teeth[teeth_array == label] = true_id
                    else:
                        true_id_teeth[teeth_array == label] = single_id[teeth_array == label]
                elif len(ids) == 1:
                    true_id = ids[0]
                else:
                    true_id = 0
                true_id_teeth[teeth_array == label] = true_id

        ### label fusion

        segmentation_result1[segmentation_result1 ==8] = 0 
        true_id_teeth[true_id_teeth!=0] += 7
        segmentation_result = np.where(segmentation_result1 != 0 , segmentation_result1, true_id_teeth)
        segmentation_result = segmentation_result.astype(np.uint8)
        labels = np.unique(segmentation_result)
        labels = sorted(labels, reverse=True)
        for label in labels:
            segmentation_result[segmentation_result == label] = self.new_old_mapping[str(label)]

        output = sitk.GetImageFromArray(segmentation_result)
        output.SetSpacing(properties_dict['sitk_stuff']['spacing'])
        output.SetOrigin(properties_dict['sitk_stuff']['origin'])
        output.SetDirection(properties_dict['sitk_stuff']['direction'])
        return output

if __name__ == '__main__':
    

    seg_al = segmentation_algorithm()
    img_paths = glob('/YourPath/*')
    out_dir = 'ResultPath'
    os.makedirs(out_dir,exist_ok=True)
    for img_path in img_paths:
        img = sitk.ReadImage(img_path)
        output = seg_al.predict(img)
        sitk.WriteImage(output , join(out_dir,os.path.basename(img_path)))
        