import math
import random

import numpy
import torch
from PIL import Image, ImageOps, ImageEnhance

max_value = 10.


def print_benchmark(model, shape):
    import os
    import onnx
    from onnx import optimizer
    from caffe2.proto import caffe2_pb2
    from caffe2.python.onnx.backend import Caffe2Backend
    from caffe2.python import core, model_helper, workspace

    inputs = torch.randn(shape, requires_grad=True)
    model(inputs)

    # export torch to onnx
    dynamic_axes = {'input0': {0: 'batch'}, 'output0': {0: 'batch'}}

    _ = torch.onnx.export(model, inputs, 'weights/model.onnx', True, False,
                          input_names=["input0"],
                          output_names=["output0"],
                          keep_initializers_as_inputs=True,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                          dynamic_axes=dynamic_axes,
                          opset_version=10)

    onnx.checker.check_model(onnx.load('weights/model.onnx'))

    # export onnx to caffe2
    onnx_model = onnx.load('weights/model.onnx')

    # Optimizer passes to perform
    passes = ['eliminate_identity',
              'eliminate_deadend',
              'eliminate_nop_dropout',
              'eliminate_nop_pad',
              'eliminate_nop_transpose',
              'eliminate_unused_initializer',
              'extract_constant_to_initializer',
              'fuse_add_bias_into_conv',
              'fuse_bn_into_conv',
              'fuse_consecutive_concats',
              'fuse_consecutive_reduce_unsqueeze',
              'fuse_consecutive_squeezes',
              'fuse_consecutive_transposes',
              'fuse_matmul_add_bias_into_gemm',
              'fuse_transpose_into_gemm',
              'lift_lexical_references',
              'fuse_pad_into_conv']
    onnx_model = optimizer.optimize(onnx_model, passes)
    caffe2_init, caffe2_predict = Caffe2Backend.onnx_graph_to_caffe2_net(onnx_model)
    caffe2_init_str = caffe2_init.SerializeToString()
    with open('weights/model.init.pb', "wb") as f:
        f.write(caffe2_init_str)
    caffe2_predict_str = caffe2_predict.SerializeToString()
    with open('weights/model.pred.pb', "wb") as f:
        f.write(caffe2_predict_str)

    # print benchmark
    model = model_helper.ModelHelper(name="model", init_params=False)

    init_net_proto = caffe2_pb2.NetDef()
    with open('weights/model.init.pb', "rb") as f:
        init_net_proto.ParseFromString(f.read())
    model.param_init_net = core.Net(init_net_proto)

    predict_net_proto = caffe2_pb2.NetDef()
    with open('weights/model.pred.pb', "rb") as f:
        predict_net_proto.ParseFromString(f.read())
    model.net = core.Net(predict_net_proto)

    model.param_init_net.GaussianFill([],
                                      model.net.external_inputs[0].GetUnscopedName(),
                                      shape=shape, mean=0.0, std=1.0)
    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net)
    workspace.BenchmarkNet(model.net.Proto().name, 5, 100, True)
    # remove temp data
    os.remove('weights/model.onnx')
    os.remove('weights/model.init.pb')
    os.remove('weights/model.pred.pb')


def add_weight_decay(model, weight_decay=1e-5):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}]


def accuracy(output, target, top_k):
    with torch.no_grad():
        max_k = max(top_k)
        batch_size = target.size(0)

        _, pred = output.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def resample():
    return random.choice((Image.BILINEAR, Image.BICUBIC))


def rotate(image, factor):
    factor = (factor / max_value) * 90

    if random.random() > 0.5:
        factor *= -1

    return image.rotate(factor, resample=resample())


def shear(image, factor):
    factor = (factor / max_value) * 0.3

    if random.random() > 0.5:
        factor *= -1
    if random.random() > 0.5:
        return image.transform(image.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), resample=resample())
    else:
        return image.transform(image.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), resample=resample())


def translate(image, factor):
    factor = (factor / max_value) * 0.5

    if random.random() > 0.5:
        factor *= -1
    if random.random() > 0.5:
        pixels = factor * image.size[0]
        return image.transform(image.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), resample=resample())
    else:
        pixels = factor * image.size[1]
        return image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), resample=resample())


def auto(image, _):
    k = random.choice((1, 2, 3))
    if k == 1:
        return ImageOps.invert(image)
    elif k == 2:
        return ImageOps.equalize(image)
    else:
        return ImageOps.autocontrast(image)


def identity(image, _):
    return image


def brightness(image, factor):
    if random.random() > 0.5:
        factor = (factor / max_value) * 1.8 + 0.1
        return ImageEnhance.Brightness(image).enhance(factor)
    else:
        factor = (factor / max_value) * 0.9

        if random.random() > 0.5:
            factor *= -1

        return ImageEnhance.Brightness(image).enhance(factor)


def color(image, factor):
    if random.random() > 0.5:
        factor = (factor / max_value) * 1.8 + 0.1
        return ImageEnhance.Color(image).enhance(factor)
    else:
        factor = (factor / max_value) * 0.9

        if random.random() > 0.5:
            factor *= -1

        return ImageEnhance.Color(image).enhance(factor)


def contrast(image, factor):
    if random.random() > 0.5:
        factor = (factor / max_value) * 1.8 + 0.1
        return ImageEnhance.Contrast(image).enhance(factor)
    else:
        factor = (factor / max_value) * 0.9

        if random.random() > 0.5:
            factor *= -1

        return ImageEnhance.Contrast(image).enhance(factor)


def sharpness(image, factor):
    if random.random() > 0.5:
        factor = (factor / max_value) * 1.8 + 0.1
        return ImageEnhance.Sharpness(image).enhance(factor)
    else:
        factor = (factor / max_value) * 0.9

        if random.random() > 0.5:
            factor *= -1

        return ImageEnhance.Sharpness(image).enhance(factor)


def solar(image, factor):
    k = random.choice((1, 2, 3))
    if k == 1:
        return ImageOps.solarize(image, int((factor / max_value) * 256))
    elif k == 2:
        return ImageOps.solarize(image, 256 - int((factor / max_value) * 256))
    else:
        lut = []
        for i in range(256):
            if i < 128:
                lut.append(min(255, i + int((factor / max_value) * 110)))
            else:
                lut.append(i)
        if image.mode in ("L", "RGB"):
            if image.mode == "RGB" and len(lut) == 256:
                lut = lut + lut + lut
            return image.point(lut)
        else:
            return image


def poster(image, factor):
    k = random.choice((1, 2, 3))
    if k == 1:
        factor = int((factor / max_value) * 4)
        if factor >= 8:
            return image
        return ImageOps.posterize(image, factor)
    elif k == 2:
        factor = 4 - int((factor / max_value) * 4)
        if factor >= 8:
            return image
        return ImageOps.posterize(image, factor)
    else:
        factor = int((factor / max_value) * 4) + 4
        if factor >= 8:
            return image
        return ImageOps.posterize(image, factor)


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        i, j, h, w = self.params(image.size)
        image = image.crop((j, i, j + w, i + h))
        return image.resize([self.size, self.size], resample())

    @staticmethod
    def params(size):
        scale = (0.08, 1.0)
        ratio = (3. / 4., 4. / 3.)
        for _ in range(10):
            target_area = random.uniform(*scale) * size[0] * size[1]
            aspect_ratio = math.exp(random.uniform(*(math.log(ratio[0]), math.log(ratio[1]))))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= size[0] and h <= size[1]:
                i = random.randint(0, size[1] - h)
                j = random.randint(0, size[0] - w)
                return i, j, h, w

        in_ratio = size[0] / size[1]
        if in_ratio < min(ratio):
            w = size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = size[1]
            w = int(round(h * max(ratio)))
        else:
            w = size[0]
            h = size[1]
        i = (size[1] - h) // 2
        j = (size[0] - w) // 2
        return i, j, h, w


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        self.num = self.num + n
        self.sum = self.sum + v * n
        self.avg = self.sum / self.num


class RandomAugment:
    def __init__(self, mean=9, sigma=0.5, n=1):
        self.n = n
        self.mean = mean
        self.sigma = sigma
        self.transform = [auto, identity,
                          rotate, shear, translate,
                          brightness, color, contrast, sharpness, solar, poster]

    def __call__(self, image):
        for transform in numpy.random.choice(self.transform, self.n):
            factor = numpy.random.normal(self.mean, self.sigma)
            factor = min(max_value, max(0., factor))

            image = transform(image, factor)
        return image
