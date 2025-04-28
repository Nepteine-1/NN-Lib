#include <iostream>
#include <gtest/gtest.h>
#include <Engine/Engine.hpp>

int main(int argc, char *argv[]) {
    #ifndef TEST_BUILD
        srand(time(NULL));
        Engine e;
        e.run();
    #else
        testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    #endif
    
    return 0;
}