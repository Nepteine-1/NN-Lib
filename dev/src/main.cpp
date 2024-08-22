#include <iostream>
#include <Engine/Engine.hpp>

int main() {
    srand(time(NULL));

    #ifndef ENABLE_TEST
        Engine e;
        e.run();
    #else
        //Run test
    #endif

    return 0;
}