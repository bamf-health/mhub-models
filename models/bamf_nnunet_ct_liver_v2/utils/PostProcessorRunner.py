"""
-------------------------------------------------
MHub - Run Module for ensembling nnUNet inference.
-------------------------------------------------
-------------------------------------------------
Author: Jithendra Kumar
Email:  jithendra.kumar@bamfhealth.com
-------------------------------------------------
"""

from mhubio.core import Instance, InstanceData
from mhubio.core import Module, IO
import SimpleITK as sitk
import numpy as np



class PostProcessorRunner(Module):

    @IO.Instance
    @IO.Input('in_data', 'nifti:mod=seg:model=nnunet', the='input segmentation')
    @IO.Output('out_data', 'bamf_processed.nrrd', 'nrrd:mod=seg:processor=bamf', data='in_data', the="keep the liver and tumor segmentation")
    def task(self, instance: Instance, in_data: InstanceData, out_data: InstanceData) -> None:

        # Log bamf runner info
        self.log("Running BamfProcessor on....")
        self.log(f" > input data:  {in_data.abspath}")
        self.log(f" > output data: {out_data.abspath}")

        img_itk = sitk.ReadImage(in_data.abspath)
        img_np = sitk.GetArrayFromImage(img_itk)
        img_out_np = np.zeros(img_np.shape)
        img_out_np[img_np == 8] = 1
        img_out_np[img_np == 9] = 2

        self.log(f"Writing tmp image to {out_data.abspath}")
        img_bamf_processed_itk = sitk.GetImageFromArray(img_out_np)
        img_bamf_processed_itk.CopyInformation(img_itk)
        sitk.WriteImage(img_bamf_processed_itk, out_data.abspath)
