from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
class nnUNetTrainer_no_mirror(nnUNetTrainer):
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        self.print_to_log_file(f'mirror_axes: {mirror_axes}') ##
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes
