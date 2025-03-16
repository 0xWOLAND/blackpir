default: build test

build:
    #!/usr/bin/env bash
    rm -rf build
    mkdir build
    cd build
    cmake ..
    make

test: build
    ./build/pir_tests

clean:
    rm -rf build
