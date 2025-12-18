#include <iostream>
#include <vector>

void stack_vs_heap_demo() {
    std::cout << "\n[Stack vs Heap]\n";

    // stack allocation
    int a = 42;
    std::cout << "stack a = " << a << "\n";

    // heap allocation (C-style, 仅用于教学)
    int* b = new int(99);
    std::cout << "heap b = " << *b << "\n";

    delete b;
}

void const_ref_ptr_demo() {
    std::cout << "\n[const / reference / pointer]\n";

    int x = 10;

    int& ref = x;           // reference
    const int& cref = x;    // const reference
    int* ptr = &x;          // pointer

    ref += 1;
    *ptr += 1;
    
    /* mistake
    cref += 1;
    */

    std::cout << "x      = " << x << "\n";
    std::cout << "ref    = " << ref << "\n";
    std::cout << "cref   = " << cref << "\n";

    std::cout << "&x     =" << &x << "\n";
    std::cout << "&ref   =" << &ref << "\n";


}
