.PHONY: all build render bench test clean
MAKEFLAGS += --no-print-directory

all: build

build/CMakeCache.txt:
	cmake -S . -B build

build: build/CMakeCache.txt
	@cmake --build build -j

render: build
	@mkdir -p out && ./build/render out

bench: build
	@./build/bench

test: build
	@ctest --test-dir build --output-on-failure

clean:
	rm -rf build out
