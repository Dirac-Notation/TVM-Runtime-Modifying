import numpy as np
import json
import os

from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te
from tvm.contrib import graph_executor
from tvm.contrib import relay_viz
import tvm.testing

batch_size = 1
num_class = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)

path_so = "compiled_lib.so"
path_params = "params.json"

if os.path.exists(path_so) and os.path.exists(path_params):
    print("lib and params exists")
    lib = tvm.runtime.load_module(path_so)
    print("lib load complete")

    with open(path_params, "r") as f:
        serialized_byte_array = json.load(f)
    print("params load complete")
else:
    # IR 형태로 구성된 model code와 그에 맞는 parameter 생성
    mod, params = relay.testing.resnet.get_workload(
        num_layers=18, batch_size=batch_size, image_shape=image_shape
    )

    serialized_byte_array = tvm.runtime.save_param_dict(params)

    # set show_meta_data=True if you want to show meta data
    # print(mod.astext(show_meta_data=False))

    print("start compile")
    opt_level = 2
    target = tvm.target.cuda(arch="sm_70")
    with tvm.transform.PassContext(opt_level=opt_level):
        lib = relay.build(mod, target)
    print("success compile")

    # save compiled lib and params
    lib.export_library(path_so)
    with open(path_params, "w") as f:
        json.dump(list(serialized_byte_array), f)
    
    # viz = relay_viz.RelayVisualizer(mod)
    # viz.render()

# create random input
dev = tvm.cuda()
np.random.seed(0)
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

# create module
print()
print("create module")
module = graph_executor.GraphModule(lib["default"](dev))
print("success module")

# set input and parameters
print()
print("load data, params")
module.set_input(0, data)
print("success data")

mode = True

if mode:
    # run
    print()
    print("start run")
    module.load_run(serialized_byte_array)
    print("success run")
else:
    module.load_params(serialized_byte_array)
    print("success params")

    # run
    print()
    print("start run")
    module.run()
    print("success run")

# get output
out = module.get_output(0, tvm.nd.empty(out_shape)).numpy()

# Print first 10 elements of output
print()
print(f"result: {out[0,:10]}")