# ikfast_pybind

this is a python binding for query all ik result for give target and return the closest one to the ref joint state if got any.

# Install
Recommand to create a venv and install pybind11.

Source the venv that you created.

Clone this package.

```sh
mkdir build && cd build
cmake ..
make
make install
```
This will build the lib and python binding. The last command will also create a python pkg and install it inside your venv.

Besides, a test executable will also be created inside the build file. You can run it to see an example test output.

```sh
cd build && ./ikfast_pybind_test
```

# Usage
After build, a python module called `panda_ikfast` will be installed in your venv.



```python
import numpy as np
import panda_ikfast
panda_ikfast.test() # This will run the same test code but use pybind, should print the same result as run the executable
target = np.eye(4,dtype=np.float64)
ref = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
success, joint_values = panda_ikfast.get_ik(target, ref)

# Or you can also provide a optional max_val and min_val as joints upper and lower bound for validity check, e.g.
#max_val = [3.14,3.14,3.14,3.14,3.14,3.14,3.14]
#min_val = [-3.14,-3.14,-3.14,-3.14,-3.14,-3.14,-3.14]
#success, joint_values = panda_ikfast.get_ik(target, ref, max_val, min_val)
```