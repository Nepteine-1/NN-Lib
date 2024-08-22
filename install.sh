#!/bin/sh
# Change the following env variable to change the name of the executable generated with cmake
export EXE_TITLE="SFML_App"

# --------------------------------------

export TARGET="$1"
export BUILD_MODE="$2"
export LIB_NAME="NeuralNetwork"

printb() {
    tput setaf $2
    tput bold
    echo $1
    tput sgr0
}

print_argument_error() {
		printb "#==== Error ====#" 1
		printb " Unknown argument - $1 $2" 9
		printb " $3" 9
		printb "#==== Stopped ====#" 1
		exit 1
}

if [ "$TARGET" = "" ] || [ "$TARGET" = "dev" ] || [ "$TARGET" = "shared" ] || [ "$TARGET" = "static" ]; then
	printb "#==== Installation ====#" 2
	cd build
	if [ "$BUILD_MODE" = "debug" ] || [ "$BUILD_MODE" = "" ]; then
		cmake . -DCMAKE_BUILD_TYPE=Debug
		BUILD_MODE="debug"
	elif [ "$BUILD_MODE" = "release" ]; then
		cmake . -DCMAKE_BUILD_TYPE=Release	
	else
		print_argument_error "Build mode : " $BUILD_MODE "Possible values [debug/release]"
	fi
	cmake --build . 
	cmake --install .
	cd ..
	mkdir bin 2>/dev/null
	mkdir bin/${BUILD_MODE} 2>/dev/null
	mv ./build/${EXE_TITLE} ./bin/${BUILD_MODE} 2>/dev/null	
	mv ./build/${LIB_NAME}.a ./bin/${BUILD_MODE} 2>/dev/null
	mv ./build/${LIB_NAME}.so ./bin/${BUILD_MODE} 2>/dev/null
	printb "#==== Finished ====#" 2
elif [ "$TARGET" = "example" ]; then
	printb "#==== Building examples ====#" 3
	cd example
	if [ "$BUILD_MODE" = "debug" ] || [ "$BUILD_MODE" = "" ]; then
		cmake . -DCMAKE_BUILD_TYPE=Debug
	elif [ "$BUILD_MODE" = "release" ]; then
		cmake . -DCMAKE_BUILD_TYPE=Release	
	fi
	cmake --build . 
	cmake --install .
	cd ..
	printb "#==== Finished ====#" 3
else
	print_argument_error "Target : " $TARGET "Possible values [dev/shared/static/example]"
fi
