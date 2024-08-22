#include <iostream>
#include <NeuralNetwork/NeuralNetwork.hpp>
#include <iomanip>

void generate_data_linear(Matrix& X_feature, Matrix& Y_class) {    
    const float a{-1.4f}, b{1.0f}, c{0.3f};
    const float zoom{500.f};
    int nb_of_class0=0;

    // Create random data
    for(int i=0; i<X_feature.col(); i++) {
        for(int j=0; j<X_feature.row(); j++) X_feature.setCoeff(j,i,randomf(0,1));

        if(X_feature.getCoeff(0,i)*a + X_feature.getCoeff(1,i)*b +c >= 0) { // Class 0
            Y_class.setCoeff(0,i,0);
        }
        else { // Class 1
            Y_class.setCoeff(0,i,1);
        }
    }
    std::cout << "data created [class 0: " << nb_of_class0 << "(" << ((float)nb_of_class0/(float)X_feature.col())*100.f <<"%)]" << std::endl;
    std::cout << std::setw(23)<< " [class 1: " << X_feature.col() - nb_of_class0 << "(" << ((float)(X_feature.col() - nb_of_class0)/(float)X_feature.col())*100.f <<"%)]" << std::endl;  
}

int main() {
    srand(time(NULL));

    NeuralNetwork my_network(2, LossFunction::BINARY_CROSS_ENTROPY);
    my_network.addLayer(8, Activation::RELU);
    my_network.addLayer(1, Activation::SIGMOID);

    Matrix X_train(2,100), Y_train(1,100);
    generate_data_linear(X_train, Y_train);

    my_network.train(X_train, Y_train, 10000, 0.3f);

    Matrix X_test(2,10), Y_test(1,10);
    generate_data_linear(X_test, Y_test);

    my_network.predict(X_test, Y_test).disp();

    return 0;
}
