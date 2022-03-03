[EfficientNet](https://arxiv.org/abs/1905.11946) implementation using PyTorch

#### Steps
* Configure `imagenet` path by changing `data_dir` in `main.py`
* `python main.py --benchmark` for model information
* `bash ./main.sh $ --train` for training model, `$` is number of GPUs
* `python main.py --test` for testing
* See `EfficientNet` class in `nets/nn.py` for different versions

#### Note
* EfficientNet-B0 achieved 77.2 % top-1 and 93.48 % top-5 after 450 epochs

```
Number of parameters: 5267540
Time per operator type:
        400.231 ms.    85.9425%. Conv
        42.3814 ms.    9.10067%. Sigmoid
        19.0129 ms.    4.08269%. Mul
        1.83499 ms.   0.394031%. AveragePool
        1.59307 ms.   0.342084%. FC
       0.636682 ms.   0.136716%. Add
      0.0058625 ms. 0.00125887%. Flatten
        465.696 ms in Total
FLOP per operator type:
        0.76907 GFLOP.    98.5601%. Conv
     0.00846444 GFLOP.    1.08476%. Mul
       0.002561 GFLOP.   0.328205%. FC
    0.000210112 GFLOP.  0.0269269%. Add
       0.780305 GFLOP in Total
Feature Memory Read per operator type:
        58.5253 MB.    53.8803%. Mul
        43.2855 MB.    39.8501%. Conv
        5.12912 MB.    4.72204%. FC
         1.6809 MB.    1.54749%. Add
        108.621 MB in Total
Feature Memory Written per operator type:
        33.8578 MB.    54.8834%. Mul
        26.9881 MB.    43.7477%. Conv
       0.840448 MB.    1.36237%. Add
          0.004 MB. 0.00648399%. FC
        61.6904 MB in Total
Parameter Memory per operator type:
        15.8248 MB.    75.5403%. Conv
          5.124 MB.    24.4597%. FC
              0 MB.          0%. Add
              0 MB.          0%. Mul
        20.9488 MB in Total
```

