import os
import cv2
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt

quant_model_path = "pytorchmodel_v2_quant.onnx"
sample_image_path = os.path.join("../data/screw/test/good", "000.png")

img = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (1024, 1024))
img = img.astype(np.float32) / 255.0
img = np.expand_dims(img, axis=(0, 1))  # Shape: (1, 1, 1024, 1024)

session = ort.InferenceSession(quant_model_path)
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: img})

for i, o in enumerate(output):
    print(f"Output {i}: shape = {o.shape}, dtype = {o.dtype}")

recon_img = output[0][0, 0]

plt.imshow(recon_img, cmap='gray')
plt.title("Quantized Autoencoder Output")
plt.axis("off")
plt.show()