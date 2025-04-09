#include <iostream>
#include <NeuralNetwork/NeuralNetwork.hpp>
#include <iomanip>

int main() {
    srand(time(NULL));

    // CREATE & TRAIN NEURAL NETWORK
    NeuralNetwork my_network(2, LossFunction::BINARY_CROSS_ENTROPY);
    my_network.addLayer(12, Activation::ELU);
    my_network.addLayer(12, Activation::ELU);
    my_network.addLayer(1, Activation::SIGMOID);

    // GENERATE DATA FOR TRAINING
    int data_number_train{250};
    Matrix X_train{Matrix(2,data_number_train)}, Y_train{Matrix(1,data_number_train)};
    generateData_Circle(X_train, Y_train, true);

    my_network.train(X_train,Y_train, 1000, 1.1f,true);
    

    // GENERATE DATA FOR PREDICTIONS
    int data_number_test{1000};
    Matrix X_test{Matrix(2,data_number_test)}, Y_test{Matrix(1,data_number_test)};
    generateData_Circle(X_test, Y_test, true);

    // PREDICT WITH NEURAL NETWORK
    Matrix pred{my_network.predict(X_test, Y_test)};
    int good_answer_num{0};
    int res{0};
    for(int i=0; i< Y_test.col(); i++) {
        if(pred.getCoeff(0,i)>=0.5) res=1;
        else res=0;
        if(res == Y_test.getCoeff(0,i)) good_answer_num++;
    }
    std::cout << "Score: " << good_answer_num << "/" << Y_test.col() << std::endl; 

    return 0;
}
