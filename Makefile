.PHONY: all build render test clean
MAKEFLAGS += --no-print-directory

all: build

build/CMakeCache.txt:
	cmake -S . -B build

build: build/CMakeCache.txt
	@cmake --build build -j

# CPU L-system demo
render: build
	@mkdir -p out && ./build/render out

test: build
	@ctest --test-dir build --output-on-failure

clean:
	rm -rf build out
