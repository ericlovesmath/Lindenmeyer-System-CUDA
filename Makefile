.PHONY: all build run test clean
MAKEFLAGS += --no-print-directory

# Treat any extra args after `make run` as opaque values, not targets.
RUN_ARGS := $(filter-out run,$(MAKECMDGOALS))
%:
	@:

all: build

build/CMakeCache.txt:
	cmake -S . -B build

build: build/CMakeCache.txt
	@cmake --build build -j

run: build
	@./build/lsystem $(RUN_ARGS)

test: build
	@ctest --test-dir build --output-on-failure

clean:
	rm -rf build
