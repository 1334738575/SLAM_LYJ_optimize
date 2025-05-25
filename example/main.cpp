#include <iostream>
#include <Optimize_LYJ.h>

int main(int argc, char *argv[])
{
    std::cout << "Optimize Version: " << OPTIMIZE_LYJ::optimize_version() << std::endl;
    OPTIMIZE_LYJ::test_optimize();
    return 0;
}