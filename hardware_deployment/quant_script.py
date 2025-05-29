import os
import numpy as np
import onnx
import cv2
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

dir_good = os.path.abspath("../data/screw/test/good")

model_path = "pytorchmodel_v2.onnx"
quant_model_path = "pytorchmodel_v2_quant.onnx"
model = onnx.load(model_path)
input_name = model.graph.input[0].name

class GrayscaleImageReader(CalibrationDataReader):
    def __init__(self, input_name, image_dir):
        self.input_name = input_name
        self.image_paths = [os.path.join(image_dir, f)
                            for f in os.listdir(image_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.iterator = self._preprocess()

    def _preprocess(self):
        for path in self.image_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (1024, 1024))
            img = img.astype(np.float32) / 255.0  # Normalize
            img = np.expand_dims(img, axis=(0, 1))  # Shape: (1, 1, 1024, 1024)
            yield {self.input_name: img}

    def get_next(self):
        return next(self.iterator, None)

reader = GrayscaleImageReader(input_name=input_name, image_dir=dir_good)

quantize_static(
    model_input=model_path,
    model_output=quant_model_path,
    calibration_data_reader=reader,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8
)

print(f"Quantized model saved to: {quant_model_path}")
