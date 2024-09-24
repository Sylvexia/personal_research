```cpp
// mycpp_library.cpp
#include <iostream>

class MyCppClass {
public:
    void greet() {
        std::cout << "Hello from C++!" << std::endl;
    }
};

// This is the wrapper function with C linkage
extern "C" {
    // C-compatible function to create a MyCppClass object
    MyCppClass* create_object() {
        return new MyCppClass();
    }

    // C-compatible function to call the greet method
    void call_greet(MyCppClass* obj) {
        obj->greet();
    }

    // C-compatible function to delete the object
    void delete_object(MyCppClass* obj) {
        delete obj;
    }
}

```