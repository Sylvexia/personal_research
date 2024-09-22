## Pybind
```bash
pip install pybind11
```

```cpp
// example.cpp
#include <pybind11/pybind11.h>

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // Optional module docstring
    m.def("add", &add, "A function that adds two numbers");
}

```

```bash
c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` example.cpp -o example`python3-config --extension-suffix`
```

```python
import example
print(example.add(2, 3))  # Outputs: 5
```

## SWIG

```bash
sudo apt-get install swig  # On Debian-based systems
```

```cpp
// example.cpp
int add(int i, int j) {
    return i + j;
}
```

```swig
// example.i
%module example
%{
extern int add(int i, int j);
%}
extern int add(int i, int j);
```

```
swig -python example.i
```

```
c++ -O2 -fPIC -I/usr/include/python3.x -c example_wrap.cxx example.cpp
c++ -shared example_wrap.o example.o -o _example.so
```

```
import example
print(example.add(2, 3))  # Outputs: 5
```