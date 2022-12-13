import logging
import numpy as np
import pathlib as Path
import SimpleITK as sitk
from typing import Tuple

logging.basicConfig(level=logging.INFO, format='%(message)s')

DEFAULT_NORM_CONFIG = {
    'type': 'min_max',
    'max_val': 255,
    'mask': None,
    'percentiles': (1, 99),
    'dtype': np.uint8
}

DEFAULT_RESIZE_CONFIG = {
    'size': (1, 1, 1),
    'interpolation': 'bicubic'
}


def min_max_norm(
    img: np.ndarray, max_val: int = None, mask: np.ndarray = None,
    percentiles: tuple = None, dtype: str = None
) -> np.ndarray:
    """
    Scales image intensities into the range [0, max_val] or [0, 2**bits]. The
    min and max values can be obtained from within a mask and using certain
    percentiles of the intensities distribution.
    Args:
        img (np.ndarray): Image to be scaled.
        max_val (int, optional): Value to scale images
            to after normalization. Defaults to None.
        mask (np.ndarray, optional): Mask to use in the normalization process.
            Defaults to None which means no mask is used.
        percentiles (tuple, optional): Tuple of percentiles to obtain min and max
            values respectively. Defaults to None which means min and max are used.
        dtype (str, optional): Output datatype
    Returns:
        np.ndarray: Scaled image with values from [0, max_val]
    """
    if mask is None:
        mask = np.ones_like(img)

    # Find min and max among the selected voxels
    if percentiles is not None:
        perc = np.percentile(img[mask != 0], list(percentiles))
        img_max = perc[1]
        img_min = perc[0]
    else:
        img_max = np.max(img[mask != 0])
        img_min = np.min(img[mask != 0])

    if max_val is None:
        # Determine possible max value according to data type
        max_val = np.iinfo(img.dtype).max

    # Normalize
    img = (img - img_min) / (img_max - img_min) * max_val
    img = np.clip(img, 0, max_val)

    # Adjust data type
    img = img.astype(dtype) if dtype is not None else img
    return img


class Preprocessor():
    def __init__(
        self,
        normalization_cfg: dict = DEFAULT_NORM_CONFIG,
        skull_stripping: bool = False,
        resize_cfg: dict = DEFAULT_RESIZE_CONFIG,
        mni_atlas: np.ndarray = None,
        mv_atlas: np.ndarray = None,
        register_atlases: bool = False,
        tissue_models: np.ndarray = None
    ) -> None:
        self.normalization_cfg = normalization_cfg
        self.skull_stripping = skull_stripping
        self.resize_cfg = resize_cfg
        self.mni_atlas = mni_atlas
        self.mv_atlas = mv_atlas
        self.register_atlases = register_atlases
        if (self.mni_atlas is not None) or (self.mv_atlas is not None):
            self.register_atlases = True
        self.tissue_models = tissue_models

    def preprocess(
        self, img: np.ndarray, brain_mask: np.ndarray = None, reg_path: Path = None
    ) -> np.ndarray:
        # Normalize image intensities
        if self.normalization_cfg is not None:
            normalization_cfg = self.normalization_cfg.copy()
            normalization_cfg['mask'] = brain_mask if brain_mask is not None else None
            if normalization_cfg['type'] == 'min_max':
                del normalization_cfg['type']
                img = min_max_norm(img, **normalization_cfg)
            else:
                raise Exception(
                    f'Normalization {self.normalization_cfg["type"]} not implemented'
                )

        # Register the atlases to the particular case.
        r_mni_atlas, r_mv_atlas = None, None
        if self.register_atlases:
            r_mni_atlas, r_mv_atlas = self.register(img, reg_path)

        # If necessary do skull stripping
        if self.skull_strip:
            img = self.skull_strip(img, brain_mask)

        # Resize img and registered atlas
        if self.resize_cfg is not None:
            img = self.resize(img, **self.resize_cfg)
            if self.register_atlases:
                r_mni_atlas = self.resize(r_mni_atlas, **self.resize_cfg)
                r_mv_atlas = self.resize(r_mv_atlas, **self.resize_cfg)

        # Get tissue models labels for the image
        tm_labels = None
        if self.tissue_models is not None:
            tm_labels = self.get_tissue_models_labels(img, brain_mask)

        return img, r_mni_atlas, r_mv_atlas, tm_labels

    def resize(
        self, img: np.ndarray, size: tuple, interpolation: str
    ) -> np.ndarray:
        # logging.warning('Resizing not implemented, returning same image')
        return img

    def skull_strip(self, img: np.ndarray,  mask: np.ndarray = None) -> np.ndarray:
        img[mask == 0] = 0
        # logging.warning('Skull Strippping not implemented, returning same image')
        return img

    def register(self, img: np.ndarray, reg_path: Path = None) -> Tuple[np.ndarray]:
        """This is a dummy function, the registration is not done.
        Just the registered images are loaded.
        """
        if reg_path is None:
            r_mni_atlas = self.register_to_img(self.mni_atlas, img)
            r_mv_atlas = self.register_to_img(self.r_mv_atlas, img)
        else:
            r_mni_atlas = sitk.GetArrayFromImage(
                sitk.ReadImage(str(reg_path).replace('.nii.gz', '_mni_atlas.nii.gz')))
            r_mv_atlas = sitk.GetArrayFromImage(
                sitk.ReadImage(str(reg_path).replace('.nii.gz', '_mv_atlas.nii.gz')))
        return r_mni_atlas, r_mv_atlas

    def register_to_img(self, atlas: np.ndarray, img: np.ndarray):
        # logging.warning(
        #     'Registration not implemented, returning moving image without modification'
        # )
        return atlas

    def get_tissue_models_labels(self, img: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
        """
        Gets a segmentation from the tissue models labels.
        Args:
            img (np.ndarray): T1 volumne.
            brain_mask (np.ndarray): Mask of the brain tissues (result of skull stripping)
        Returns:
            _type_: _description_
        """
        # Consider only the region inside the brain mask
        t1_vector = img[brain_mask == 255].flatten()
        # Set the probability values based on the tissue models (posterior probabilities)
        n_classes = self.tissue_models.shape[0]
        prob_map = np.zeros((n_classes, len(t1_vector)))
        for c in range(n_classes):
            prob_map[c, :] = self.tissue_models[c, t1_vector]
        # MAP label assignment
        prob_map = np.argmax(prob_map, axis=0)
        # Reconstruct a single 3d volume with the segmentation results.
        tm_labels_vol = brain_mask.flatten()
        tm_labels_vol[tm_labels_vol == 255] = prob_map + 1
        tm_labels_vol = tm_labels_vol.reshape(img.shape)
        return tm_labels_vol
