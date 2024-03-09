import torch
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, Resizer, Normalizer
from retinanet import coco_eval

def main(args=None):
    model_path ='/kaggle/input/retina_net/pytorch/model/1/coco_resnet_50_map_0_335_state_dict.pt'
    retinanet = model.resnet50(num_classes=80, pretrained=True)
    retinanet.load_state_dict(torch.load(model_path))
    example_input = torch.randn(1, 3, 600,800)
    onnx_path = "detr.onnx"
    torch.onnx.export(retinanet,                               # model being run
                  (example_input, ),                  # model input (or a tuple for multiple inputs)
                  onnx_path,                           # where to save the model (can be a file or file-like object)
                  export_params=True,                  # store the trained parameter weights inside the model file
                  opset_version=12,                    # the ONNX version to export the model to
                  do_constant_folding=True,            # whether to execute constant folding for optimization
                  input_names=["input"],               # the model's input names
                  output_names=["output"],             # the model's output names
                  dynamic_axes={"input": {0: "batch_size"},  # variable length axes
                                "output": {0: "batch_size"}},verbose=True)
    



if __name__ == '__main__':
    main()
