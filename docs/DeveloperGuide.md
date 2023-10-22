# Developer's Guide for Speech-to-Text Models

**Converting Your Own Model**

To begin, please check if your model can be successfully converted into ONNX format without any special treatment using `ASRModelExport`. It's important to note that if your model contains operations not supported by PyTorch, you may encounter errors. In such cases, follow these steps to successfully convert your model:

1. Create a new class that is ONNX-compatible, excluding any unsupported operations.

2. Integrate your newly created class into the `espnet_onnx.export.convert_map.yml` file. This file will help ESPnet-ONNX identify the conversion between incompatible and compatible classes. Here's an example of how to add your class to the YAML file:

   ```yaml
   asr:
    ...

    # Add your new class here
    - from: <incompatible class>
      to: <compatible class>
   ```

3. After adding your class to the `convert_map.yml` file, check if you can successfully convert your model into the ONNX format. ESPnet-ONNX will automatically identify the incompatible classes and replace them with the compatible ones, ensuring a seamless conversion process.
