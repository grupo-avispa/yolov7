from openvino.tools import mo
from openvino.runtime import serialize

model = mo.convert_model('best.onnx')
serialize(model, 'best.xml')
