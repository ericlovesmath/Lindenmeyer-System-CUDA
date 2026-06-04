.PHONY: all build render playground bench profile test clean
MAKEFLAGS += --no-print-directory

all: build

build/CMakeCache.txt:
	cmake -S . -B build

build: build/CMakeCache.txt
	@cmake --build build -j

render: build
	@mkdir -p out && ./build/render out

playground: build
	@mkdir -p out && ./build/playground

bench: build
	@./build/bench

profile: build
	@scripts/ncu.sh

test: build
	@ctest --test-dir build --output-on-failure

clean:
	rm -rf build out
