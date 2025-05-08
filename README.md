# NN-Library
A library dedicated to the creation and training of neural networks, made from scratch.

## Required packages
```bash
sudo apt install cmake
sudo apt install libsfml-dev
```

## Installation
Use the following command pattern to launch the installation.
```bash
sh install.sh target build_mode
```

__target__ must be replaced with one of the following values: [dev/shared/static/example]
* __target = dev__ by default if not defined

__build_mode__ must be replaced as well with : [debug/release]
* __build_mode = debug__ by default if not defined

For instance, if I choose to build a shared version of the library in release mode:
```bash
sh install.sh shared release
```

If I choose to build the sources files in debug mode to get an executable for developpement purpose :
```bash
sh install.sh dev debug
# sh install.sh debug - Valid
# sh install.sh - Valid
```

In the case where __target=example__, build the static library in release mode before using this target because the example project (in 'example' directory) use it.
```bash
sh install.sh static release
sh install.sh example
```

Then the binairies will appear in the "bin" directory.

## Documentation

### Create a neural network
Create a new neural network.
it is possible to choose the loss function. Currently, only BINARY_CROSS_ENTROPY is available.
In the following example, an empty neural network with 2 entry neurons is created:
```cpp
// NeuralNetwork(const int number_of_features, const LossFunction loss_type);
nn = new NeuralNetwork(2, LossFunction::BINARY_CROSS_ENTROPY);
```

### Add a layer of neurons
Create a new layer of neurons, it is possible to choose the activation function of each layer.
In the following example, a neural network with 3 layers is created:
```cpp
// void addLayer(const int nb_neuron, const Activation& activation_type = Activation::SIGMOID);
// enum Activation {SIGMOID, RELU, ELU};
nn->addLayer(12, Activation::ELU);
nn->addLayer(12, Activation::ELU);
nn->addLayer(1, Activation::SIGMOID);
```

### Generate training/test data
Generate data following a pattern.
```cpp
// void generateData_Linear(Matrix& X_feature,Matrix& Y_class);
// void generateData_Circle(Matrix& X_feature,Matrix& Y_class);
// void generateData_Balanced(Matrix& X_feature,Matrix& Y_class);
// void generateData_3Class(Matrix& X_feature,Matrix& Y_class);
int data_number_train{250};
Matrix X_train{Matrix(2,data_number_train)}, Y_train{Matrix(1,data_number_train)};
generateData_Circle(X_train, Y_train);
```

### Train
Train the neural network with the given train data.
```cpp
// void train(const Matrix& X_train, const Matrix& Y_train, const int epoch=1, const float learning_rate=1.0f, const bool show_result=true);
// 'show_result' is a deprecated argument
nn->train(X_train,Y_train, 1000, 1.1f,true);
```

### Predict
Make predictions with the given test data.
```cpp
// Matrix predict(const Matrix& X_test, const Matrix& Y_test);
int data_number_test{250};
Matrix X_test{Matrix(2,data_number_test)}, Y_test{Matrix(1,data_number_test)};
generateData_Circle(X_test, Y_test);

nn->predict(X_test, Y_test);
```

### Clear the neural network
Remove all the layers of the neural network.
```cpp
// void clear(void);
nn->clear();
```

## Build with Docker
It is now possible to compile the library into a Docker container without installing the dependencies on your machine.

Docker must be installed first, then type the following commands:

```bash
docker build -t nn-lib_image:v1.0 .
docker run -ti --rm -u nn-lib nn-lib_image:v1.0
```

The docker image configured in the Dockerfile will compile the library in static mode and run the 'example' project.
