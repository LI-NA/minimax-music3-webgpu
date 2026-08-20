from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


root = Path('test-fixtures')
root.mkdir(exist_ok=True)

external = helper.make_tensor('weight', TensorProto.FLOAT, [1], [2.0])
model = helper.make_model(
    helper.make_graph([helper.make_node('Add', ['input', 'weight'], ['output'])], 'external-add', [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1])], [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1])], [external]),
    opset_imports=[helper.make_opsetid('', 13)],
)
onnx.save_model(model, root / 'external-add.onnx', save_as_external_data=True, all_tensors_to_one_file=True, location='external-add.bin', size_threshold=0)

packed = np.stack((np.full((1, 64), 0x99, dtype=np.uint8), np.full((1, 64), 0xAA, dtype=np.uint8)))
scales = np.ones((2, 1), dtype=np.float16)
matmul = helper.make_model(
    helper.make_graph([helper.make_node('MatMulNBits', ['input', 'weight', 'scales'], ['output'], domain='com.microsoft', bits=4, block_size=128, accuracy_level=4, K=128, N=2)], 'q4-known-answer', [helper.make_tensor_value_info('input', TensorProto.FLOAT16, [1, 128])], [helper.make_tensor_value_info('output', TensorProto.FLOAT16, [1, 2])], [numpy_helper.from_array(packed, 'weight'), numpy_helper.from_array(scales, 'scales')]),
    opset_imports=[helper.make_opsetid('', 13), helper.make_opsetid('com.microsoft', 1)],
)
onnx.save(matmul, root / 'matmul-nbits.onnx')
