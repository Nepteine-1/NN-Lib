#include <NeuralNetwork/NeuralNetwork.hpp>

#ifdef DEBUG
    #define DEBUG_MSG(str) do { std::cout << "NeuralNetwork::" << str << std::endl; } while( false )
    #define DEBUG_MSG_S(str) do { std::cout << "NeuralNetwork::" << str; } while( false )
    #define DEBUG_MSG_D() do { std::cout << " - DONE"<< std::endl; } while( false )
#else
    #define DEBUG_MSG(str) do { } while ( false )
    #define DEBUG_MSG_S(str) do { } while( false )
    #define DEBUG_MSG_D() do { } while( false )
#endif


NeuralNetwork::NeuralNetwork(const int nb_features, const LossFunction loss_type) : nb_features(nb_features), m_lossEvo(0,0, 0.0f) {
    setLossFunction(loss_type);
}

NeuralNetwork::~NeuralNetwork() {}

void NeuralNetwork::addLayer(const int nb_neuron, const Activation& activation_type) {
    DEBUG_MSG("ADD_LAYER(" << nb_neuron << " neuron(s)), activation type " << ((activation_type ==0) ? "SIGMOID" : (activation_type == 1) ? "RELU" : "ELU") << ")");
    if(m_bias.size()==0 && m_weights.size()==0) {
        initWeights(activation_type, nb_neuron, nb_features);
        initBias(activation_type, nb_neuron, 1);
        
    } else {
        initWeights(activation_type, nb_neuron, m_weights[m_weights.size()-1].row());
        initBias(activation_type, nb_neuron, 1);
    }
    m_activ_func.push_back(activation_type);
}

void  NeuralNetwork::setNbFeatures(int nb_features) {
    DEBUG_MSG("START_CHANGE_FEATURES");
    std::vector<std::pair<int, Activation>> nbNeuronsEachLayer;
    for(int i=0; i<m_bias.size();i++) nbNeuronsEachLayer.push_back(std::pair<int, Activation>(m_bias[i].row(), m_activ_func[i]));
    this->clear();
    this->nb_features = nb_features;
    for(int i=0; i<nbNeuronsEachLayer.size();i++) addLayer(nbNeuronsEachLayer[i].first, nbNeuronsEachLayer[i].second);
    DEBUG_MSG("FINISH_CHANGE_FEATURES(" << this->nb_features << ")");
}

void NeuralNetwork::initWeights(const Activation& activation_type, int nb_lines, int nb_columns) {
    std::random_device rd;     // Only used once to initialise (seed) engine
    std::mt19937 rng(rd());    // Random-number engine

    Matrix res{ Matrix(nb_lines, nb_columns, 0.0f) };
    if(activation_type == Activation::SIGMOID) {
        DEBUG_MSG_S("INITIALIZE_WEIGHTS(Xavier Initialization method)");
        float min_range{-1.0f/sqrtf(nb_columns)}, max_range{1.0f/sqrtf(nb_columns)};
        std::uniform_real_distribution<float> uni(min_range,max_range); 
        for(int i=0; i<nb_lines; i++) {
            for(int j=0; j<nb_columns; j++) {   
                res.setCoeff(i,j, uni(rng));
            }
        }
        DEBUG_MSG_D();
    } else if (activation_type == Activation::ELU or activation_type == Activation::RELU) {
        DEBUG_MSG_S("INITIALIZE_WEIGHTS(He Initialization method)");
        float max_range{sqrtf(2.0f/float(nb_columns))};
        std::uniform_real_distribution<float> uni(0.0f,max_range); 
        for(int i=0; i<nb_lines; i++) {
            for(int j=0; j<nb_columns; j++) {
                res.setCoeff(i,j, uni(rng));
            }
        }
        DEBUG_MSG_D();
    }
    // res.disp();
    m_weights.push_back(res);
}

void NeuralNetwork::initBias(const Activation& activation_type, int nb_lines, int nb_columns) {
    Matrix res{ Matrix(nb_lines, nb_columns, 0.01f) };
    m_bias.push_back(res);
}

void NeuralNetwork::clear(void) {
    DEBUG_MSG("CLEAR_NETWORK");
    DEBUG_MSG("BIAS_CAPACITY_BEFORE_CLEAN:" << m_bias.capacity());
    std::vector<Matrix>().swap(m_bias);
    DEBUG_MSG("BIAS_CAPACITY:" << m_bias.capacity());
    DEBUG_MSG("WEIGHT_CAPACITY_BEFORE_CLEAN:" << m_weights.capacity());
    std::vector<Matrix>().swap(m_weights);
    DEBUG_MSG("WEIGHT_CAPACITY:" << m_weights.capacity());
    DEBUG_MSG("ACTIV_CAPACITY_BEFORE_CLEAN:" << m_activ_func.capacity());
    std::vector<Activation>().swap(m_activ_func);
    DEBUG_MSG("ACTIV_CAPACITY:" << m_activ_func.capacity());
    DEBUG_MSG("CLEAR_NETWORK_DONE");
}

void NeuralNetwork::train(const Matrix& X_train, const Matrix& Y_train, const int epoch, const float learning_rate, const bool show_result) {
    // Creating vector to store activation matrix and z Matrix
    m_trainPercent = 0;
    std::vector<Matrix> activ;
    std::vector<Matrix> function_z;
    activ.push_back(X_train);  
    function_z.push_back(X_train);  

    float m = 1.f/(double)X_train.col();

    activ.reserve(m_weights.size()+1);
    function_z.reserve(m_weights.size()+1);
    for(int i=0; i<m_weights.size(); i++) {
        activ.push_back(Matrix(m_weights[i].row(),X_train.col()));
        function_z.push_back(Matrix(m_weights[i].row(),X_train.col()));
    }

    // Starting training process
    m_lossEvo = Matrix(Y_train.row(), epoch/m_getLossResultsEach, 0.0f);
    m_epochEvo.clear();
    m_accuEvo.clear();
    m_epochEvo.reserve(epoch/m_getLossResultsEach);
    m_accuEvo.reserve(epoch/m_getLossResultsEach);
    for(int iter=0; iter<epoch; iter++) {
        m_trainPercent=((float)iter/(float)epoch)*100;
        // Forward propagation
        for(int i=0; i<m_weights.size(); i++) {
            function_z[i+1] = m_weights[i]*activ[i];
            function_z[i+1].merge(m_bias[i]);
            activ[i+1] = function_z[i+1];   
            setActivationFunction(m_activ_func[i]);
            activ[i+1] = (*this.*activation)(activ[i+1]);
        }

        // Calc lost function
        Matrix glob_loss_function= (*this.*Loss)(activ[activ.size()-1], Y_train);
        Matrix res = Matrix(Y_train.row(), 1);
        for(int i=0; i<glob_loss_function.row();i++) {
            float add = 0.f;
            for(int j=0; j<glob_loss_function.col();j++) { 
                add += glob_loss_function.getCoeff(i,j);
            } 
            res.setCoeff(i,0,add);
        }   
        res*(-m);

        // Update loss, epoch and accuracy record
        if(iter%m_getLossResultsEach == 0 ) {
            for(int i=0; i<res.row(); i++) {
                m_lossEvo.setCoeff(i, iter/m_getLossResultsEach, res.getCoeff(i,0));
            }
            m_epochEvo.push_back(iter);
            m_accuEvo.push_back(1.0f - calculateAccu(activ[activ.size()-1],Y_train));
        }

        // SHOW DEBUG INFO
        #ifdef DEBUG
            res.disp();
            DEBUG_MSG("Train " << (float(iter)/float(epoch))*100.f << "%");
        #endif

        // Back propagation
        setActivationFunction(m_activ_func[m_activ_func.size()-1]);
        Matrix dZ = (*this.*LossDerivate)(activ[activ.size()-1], Y_train);
        dZ = Hadamard(dZ, (*this.*activationDerivate)(function_z[function_z.size()-1]));
        for(int i=activ.size()-1; i>0; i--) {
            Matrix dW{dZ};
            dW*m;
            dW = dW*(activ[i-1].transposee());

            Matrix dB{SumOnCol(dZ)};
            dB * m;

            if(i>1) {
                setActivationFunction(m_activ_func[i-2]);
                dZ = m_weights[i-1].transposee() * dZ;
                dZ = Hadamard(dZ, (*this.*activationDerivate)(function_z[i-1]));
            }

            dW * (-learning_rate);
            dB * (-learning_rate);
            
            m_weights[i-1] = m_weights[i-1] + dW;
            m_bias[i-1] = m_bias[i-1] + dB;
        }
    }

    DEBUG_MSG("FINISH_TRAINING");
}

Matrix NeuralNetwork::predict(const Matrix& X_test, const Matrix& Y_test) {
    DEBUG_MSG_S("PREDICTIONS");
    Matrix predict {X_test};
    for(int i=0; i<m_weights.size(); i++) {
            predict = m_weights[i]*predict;
            predict.merge(m_bias[i]);
            setActivationFunction(m_activ_func[i]);
            predict = (*this.*activation)(predict);
    }
    //X_test.disp();
    //predict.disp();
    //Y_test.disp();
    DEBUG_MSG_D();
    return predict;
}

void NeuralNetwork::setLossFunction(LossFunction type) {
    switch(type) {
        case LossFunction::BINARY_CROSS_ENTROPY:
            Loss = &NeuralNetwork::LogLoss;
            LossDerivate = &NeuralNetwork::LogLossDerivate;
            DEBUG_MSG( "SET_LOSS_FUNCTION" << "(LossFunction::BINARY_CROSS_ENTROPY)");
            break;
    }
}

void NeuralNetwork::setLossResultsEach(int value) { m_getLossResultsEach = value; }

Matrix& NeuralNetwork::getLossResult(void) { return m_lossEvo; }

std::vector<int>& NeuralNetwork::getEpochResult(void) { return m_epochEvo; }

std::vector<float>& NeuralNetwork::getAccuResult(void) { return m_accuEvo; }

int NeuralNetwork::getTrainPercent(void) { return m_trainPercent; }

float NeuralNetwork::calculateAccu(Matrix const& Y, Matrix const& Y_train) {
    float accu=0;
    for(int i=0; i<Y.row(); i++) {
        float temp = 0;
        for(int j=0; j<Y.col(); j++) {
            temp+= abs(Y_train.getCoeff(i,j) - Y.getCoeff(i,j));
        }
        accu += temp/(float)Y.col();
    }
    return accu/(float)Y.row();
}

void NeuralNetwork::setActivationFunction(Activation type) {
    switch(type) {
        case Activation::SIGMOID:
            activation = &NeuralNetwork::Sigmoid;
            activationDerivate = &NeuralNetwork::SigmoidDerivate;
            break;
        
        case Activation::ELU:
            activation = &NeuralNetwork::ELU;
            activationDerivate = &NeuralNetwork::ELUDerivate;
            break;
        
        case Activation::RELU:
            activation = &NeuralNetwork::RELU;
            activationDerivate = &NeuralNetwork::RELUDerivate;
            break;
    }
}

//////////////
// LOSS FUNCTION AND DERIVATE
//////////////

// Apply Log loss on matrix A and  Y
// NOTE: dim(A) = dim(Y)
Matrix NeuralNetwork::LogLoss(const Matrix& A, const Matrix& Y) {
    float eps = 1e-8f;
    Matrix res = Matrix(A.row(), A.col(),0.f);
    for(int i=0; i<A.row(); i++) {
        for(int j=0; j<A.col(); j++) {
            res.setCoeff(i,j, Y.getCoeff(i,j) * log(A.getCoeff(i,j)+eps)+ (1.f-Y.getCoeff(i,j))*log(1.f-A.getCoeff(i,j)+eps));
        }   
    }
    return res;
}

// Apply Log loss derivate on matrix A and  Y
// NOTE: dim(A) = dim(Y)
Matrix NeuralNetwork::LogLossDerivate(const Matrix& A, const Matrix& Y) {
    float eps = 1e-8f;
    Matrix res {Matrix(A.row(), A.col(),0.f)};
    for(int i=0; i<A.row(); i++) {
        for(int j=0; j<A.col(); j++) {
            res.setCoeff(i,j, -Y.getCoeff(i,j)/(A.getCoeff(i,j)+eps) + (1.f-Y.getCoeff(i,j))/(1.f-A.getCoeff(i,j)+eps));
        }   
    }
    return res;
}

//////////////
// ACTIVATION FUNCTION AND DERIVATE
//////////////

Matrix NeuralNetwork::Sigmoid(const Matrix& Z) {
    Matrix res {Matrix(Z.row(), Z.col(),0.f)};
    for(int i=0; i<Z.row(); i++) {
        for(int j=0; j<Z.col(); j++) {
            res.setCoeff(i,j, 1.f / (1.f + exp(-1.f * Z.getCoeff(i,j))));
        }
    }
    return res;
}

Matrix NeuralNetwork::SigmoidDerivate(const Matrix& Z) {
    Matrix res {Matrix(Z.row(), Z.col(),0.f)};
    for(int i=0; i<Z.row(); i++) {
        for(int j=0; j<Z.col(); j++) {
            float sigmo = 1.f / (1.f + exp(-1.f * Z.getCoeff(i,j)));
            res.setCoeff(i,j, sigmo * (1.f - sigmo));
        }
    }
    return res;
}

Matrix NeuralNetwork::ELU(const Matrix& Z) {
    float alpha=1.f;
    Matrix res {Z};
    for(int i=0; i<Z.row(); i++) {
        for(int j=0; j<Z.col(); j++) {
            if(Z.getCoeff(i,j)>0.f) res.setCoeff(i,j, Z.getCoeff(i,j));
            else res.setCoeff(i,j,alpha*(exp(Z.getCoeff(i,j))-1.f));
        }
    }
    return res;
}

Matrix NeuralNetwork::ELUDerivate(const Matrix& Z) {
    float alpha=1.f;
    Matrix res {Z};
    for(int i=0; i<Z.row(); i++) {
        for(int j=0; j<Z.col(); j++) {
            if(Z.getCoeff(i,j)>=0.f) res.setCoeff(i,j, 1.f);
            else res.setCoeff(i,j,alpha*exp(Z.getCoeff(i,j)));
        }
    }
    return res;
}

Matrix NeuralNetwork::RELU(const Matrix& Z) {
    Matrix res {Z};
    for(int i=0; i<Z.row(); i++) {
        for(int j=0; j<Z.col(); j++) {
            if(Z.getCoeff(i,j)>0.f) res.setCoeff(i,j, Z.getCoeff(i,j));
            else res.setCoeff(i,j,0.f);
        }
    }
    return res;
}

Matrix NeuralNetwork::RELUDerivate(const Matrix& Z) {
    Matrix res {Z};
    for(int i=0; i<Z.row(); i++) {
        for(int j=0; j<Z.col(); j++) {
            if(Z.getCoeff(i,j)>0.f) res.setCoeff(i,j, 1.f);
            else res.setCoeff(i,j,0.f);
        }
    }
    return res;
}

// This function generate linearly separable data with 2 class
// X_feature max dim : row=2, col= nb_of_data
// Y_class max dim: row=1, col=nb_of_data
void generateData_Linear(Matrix& X_feature,Matrix& Y_class) {
    DEBUG_MSG_S("GENERATE_DATA(Linear)");
    const float a{-1.4f}, b{1.0f}, c{0.3f};
    const float zoom{500.f};
    int nb_of_class0=0;

    // Create random data
    for(int i=0; i<X_feature.col(); i++) {
        for(int j=0; j<X_feature.row(); j++) X_feature.setCoeff(j,i,randomf(0,1));


        if(X_feature.getCoeff(0,i)*a + X_feature.getCoeff(1,i)*b +c >= 0) { // Class 0
            Y_class.setCoeff(0,i,0);
            nb_of_class0++;
        }
        else { // Class 1
            Y_class.setCoeff(0,i,1);
        }
    }
    /*std::cout << "data created [class 0: " << nb_of_class0 << "(" << ((float)nb_of_class0/(float)X_feature.col())*100.f <<"%)]" << std::endl;
    std::cout << std::setw(23)<< " [class 1: " << X_feature.col() - nb_of_class0 << "(" << ((float)(X_feature.col() - nb_of_class0)/(float)X_feature.col())*100.f <<"%)]" << std::endl; */ 
    DEBUG_MSG_D();
}

// This function generate non linearly separable data with 2 class
// X_feature max dim : row=2, col= nb_of_data
// Y_class max dim: row=1, col=nb_of_data
void generateData_Circle(Matrix& X_feature,Matrix& Y_class) {
    DEBUG_MSG_S("GENERATE_DATA(Circle)");
    const float r{0.4f}, x{0.5f}, y{0.5f};
    const float zoom{500.f};
    int nb_of_class0=0;

    // Create random data
    for(int i=0; i<X_feature.col(); i++) {
        for(int j=0; j<X_feature.row(); j++) X_feature.setCoeff(j,i,randomf(0,1));

        if((X_feature.getCoeff(0,i)-x)*(X_feature.getCoeff(0,i)-x) + (X_feature.getCoeff(1,i)-y)*(X_feature.getCoeff(1,i)-y) > r*r) { // Class 0
            Y_class.setCoeff(0,i,0);
            nb_of_class0++;
        }
        else { // Class 1
            Y_class.setCoeff(0,i,1);
        }
    }
    /*std::cout << "data created [class 0: " << nb_of_class0 << "(" << ((float)nb_of_class0/(float)X_feature.col())*100.f <<"%)]" << std::endl;
    std::cout << std::setw(23)<< " [class 1: " << X_feature.col() - nb_of_class0 << "(" << ((float)(X_feature.col() - nb_of_class0)/(float)X_feature.col())*100.f <<"%)]" << std::endl; */
    DEBUG_MSG_D();
}

// This function generate non linearly separable data with 2 class
// X_feature max dim : row=2, col= nb_of_data
// Y_class max dim: row=1, col=nb_of_data
void generateData_Balanced(Matrix& X_feature,Matrix& Y_class) {
    DEBUG_MSG_S("GENERATE_DATA(Balanced)");
    const float r{0.4f}, x{0.5f}, y{0.5f};
    const float zoom{500.f};
    int nb_of_class0=0;

    // Create random data
    for(int i=0; i<X_feature.col(); i++) {
        for(int j=0; j<X_feature.row(); j++) X_feature.setCoeff(j,i,randomf(0,1));

        if(X_feature.getCoeff(0,i) < 0.5 && X_feature.getCoeff(1,i) < 0.5 || X_feature.getCoeff(0,i) > 0.5 && X_feature.getCoeff(1,i) > 0.5){ // Class 0
            Y_class.setCoeff(0,i,0);
            nb_of_class0++;
        }
        else { // Class 1
            Y_class.setCoeff(0,i,1);
        }
    }
    /*std::cout << "data created [class 0: " << nb_of_class0 << "(" << ((float)nb_of_class0/(float)X_feature.col())*100.f <<"%)]" << std::endl;
    std::cout << std::setw(23)<< " [class 1: " << X_feature.col() - nb_of_class0 << "(" << ((float)(X_feature.col() - nb_of_class0)/(float)X_feature.col())*100.f <<"%)]" << std::endl;*/  
    DEBUG_MSG_D();
}

// This function generate non linearly separable data with 3 class
// X_feature max dim : row=2, col= nb_of_data
// Y_class max dim: row=3, col=nb_of_data
void generateData_3Class(Matrix& X_feature,Matrix& Y_class) {
    DEBUG_MSG_S("GENERATE_DATA(3Class)");
    int nb_of_class0=0;
    int nb_of_class1=0;
    int nb_of_class2=0;

    // Create random data
    for(int i=0; i<X_feature.col(); i++) {
        for(int j=0; j<X_feature.row(); j++) X_feature.setCoeff(j,i,randomf(0,1));

        if(X_feature.getCoeff(0,i) > 0.5 && X_feature.getCoeff(1,i) > 0.5) { // Class 0
            Y_class.setCoeff(0,i,1);
            Y_class.setCoeff(1,i,0);
            Y_class.setCoeff(2,i,0);
            nb_of_class0++;
        }
        else if (X_feature.getCoeff(0,i) <= 0.5 && X_feature.getCoeff(1,i) > 0.5) { // Class 1
            Y_class.setCoeff(0,i,0);
            Y_class.setCoeff(1,i,1);
            Y_class.setCoeff(2,i,0);
            nb_of_class1++;
        }
        else if (X_feature.getCoeff(1,i) <= 0.5) {
            Y_class.setCoeff(0,i,0);
            Y_class.setCoeff(1,i,0);
            Y_class.setCoeff(2,i,1);
            nb_of_class2++;
        }
    }
    /*std::cout << "data created [class 0: " << nb_of_class0 << "(" << ((float)nb_of_class0/(float)X_feature.col())*100.f <<"%)]" << std::endl;
    std::cout << std::setw(23)<< " [class 1: " << nb_of_class1 << "(" << ((float)nb_of_class1/(float)X_feature.col())*100.f <<"%)]" << std::endl;  
    std::cout << std::setw(23)<< " [class 2: " << nb_of_class2 << "(" << ((float)nb_of_class2/(float)X_feature.col())*100.f <<"%)]" << std::endl;*/  
    DEBUG_MSG_S("GENERATE_DATA(3Class)");
}

void saveData(const Matrix& train_X, const Matrix& train_Y, const Matrix& test_X, const Matrix& test_Y, std::string outputFileName) {
    DEBUG_MSG("SAVE DATA IN FILE: " << outputFileName);
    std::ofstream file (outputFileName);

    file << train_X.row() << " " << train_X.col() << std::endl;
    for(int i=0; i< train_X.row(); i++) {
        for(int j=0; j< train_X.col(); j++) {
            file << train_X.getCoeff(i,j) << " ";
        }
        file << std::endl;
    }

    file << train_Y.row() << " " << train_Y.col() << std::endl;
    for(int i=0; i< train_Y.row(); i++) {
        for(int j=0; j< train_Y.col(); j++) {
            file << train_Y.getCoeff(i,j) << " ";
        }
        file << std::endl;
    }

    file << test_X.row() << " " << test_X.col() << std::endl;
    for(int i=0; i< test_X.row(); i++) {
        for(int j=0; j< test_X.col(); j++) {
            file << test_X.getCoeff(i,j) << " ";
        }
        file << std::endl;
    }

    file << test_Y.row() << " " << test_Y.col() << std::endl;
    for(int i=0; i< test_Y.row(); i++) {
        for(int j=0; j< test_Y.col(); j++) {
            file << test_Y.getCoeff(i,j) << " ";
        }
        file << std::endl;
    }

    file.close();
    DEBUG_MSG("SAVE DATA - DONE");
}

void loadData(Matrix& train_X, Matrix& train_Y, Matrix& test_X, Matrix& test_Y, std::string inputFileName) {
    DEBUG_MSG("LOAD DATA FROM FILE: " << inputFileName);
    std::ifstream file (inputFileName);
    if(file) {
        double row, col;

        file >> row >> col;
        train_X = Matrix(row,col, 0.0f);
        std::string line;
        std::getline(file, line);
        for(int i=0; i< row; i++) {
            std::getline(file, line);
            std::istringstream data(line);
            double data_value;
            for(int j=0; j< col; j++) {
                data>>data_value;
                train_X.setCoeff(i,j, data_value);
            }
        } 

        file >> row >> col;
        train_Y = Matrix(row,col, 0.0f);
        std::getline(file, line);
        for(int i=0; i< row; i++) {
            std::getline(file, line);
            std::istringstream data(line);
            double data_value;
            for(int j=0; j< col; j++) {
                data>>data_value;
                train_Y.setCoeff(i,j, data_value);
            }
        } 

        file >> row >> col;
        test_X = Matrix(row,col, 0.0f);
        std::getline(file, line);
        for(int i=0; i< row; i++) {
            std::getline(file, line);
            std::istringstream data(line);
            double data_value;
            for(int j=0; j< col; j++) {
                data>>data_value;
                test_X.setCoeff(i,j, data_value);
            }
        } 

        file >> row >> col;
        test_Y = Matrix(row,col, 0.0f);
        std::getline(file, line);
        for(int i=0; i< row; i++) {
            std::getline(file, line);
            std::istringstream data(line);
            double data_value;
            for(int j=0; j< col; j++) {
                data>>data_value;
                test_Y.setCoeff(i,j, data_value);
            }
        } 
        DEBUG_MSG("LOAD DATA - DONE");
    } else std::cout << "file not found: " << inputFileName << std::endl;
    
}