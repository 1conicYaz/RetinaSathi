from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from compute.auxiliary_runtime import AuxiliaryRuntime


class FakeSession:
    def __init__(self,*outputs:np.ndarray)->None:self.outputs=outputs
    def run(self,_names,_inputs):return self.outputs


class AuxiliaryRuntimeTests(unittest.TestCase):
    def test_optional_models_produce_separate_experimental_outputs(self)->None:
        runtime=object.__new__(AuxiliaryRuntime)
        lesion=np.full((1,4,16,16),-5,dtype=np.float32);lesion[:,:,6:10,6:10]=5
        vessel=np.full((1,1,16,16),-5,dtype=np.float32);vessel[:,:,4:12,7:9]=5
        runtime.lesion_session=FakeSession(lesion);runtime.lesion_manifest={"input_size":16,"threshold":0.5,"model_version":"lesion-test","classes":["microaneurysms","haemorrhages","hard_exudates","soft_exudates"]}
        runtime.vessel_session=FakeSession(vessel);runtime.vessel_manifest={"input_size":16,"threshold":0.5,"model_version":"vessel-test"}
        runtime.localization_session=FakeSession(np.asarray([[.25,.5,.65,.52]],dtype=np.float32),np.full((1,4),-4,dtype=np.float32));runtime.localization_manifest={"input_size":16,"model_version":"location-test","disc_error_scale":0.2,"fovea_error_scale":0.2}
        image=Image.new("RGB",(32,24),(130,70,35));tensor=lambda _image,size,_settings:np.zeros((1,3,size,size),dtype=np.float32)
        structures,lesions=runtime.analyze(image,tensor)
        self.assertEqual(structures["status"],"ready");self.assertEqual(lesions["status"],"ready")
        self.assertTrue(str(lesions["overlay_image"]).startswith("data:image/jpeg;base64,"))
        self.assertEqual(len(lesions["items"]),4);self.assertGreater(structures["optic_disc"]["confidence"],0)


if __name__=="__main__":unittest.main()
