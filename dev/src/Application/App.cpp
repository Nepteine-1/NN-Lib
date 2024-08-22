#include <Application/App.hpp>

#define log(str) std::cout << str << std::endl;

App::App() {
    EventHandler::getEventHandler()->addKeyBoardObserver(this);
    m_data_coord = sf::VertexArray(sf::Quads,4);
    m_frontier = sf::VertexArray(sf::Lines, 2);

    // Create NeuralNetwork
    nn = new NeuralNetwork(2, LossFunction::BINARY_CROSS_ENTROPY); // Initialize the network with an input layer with 2 entries
    nn->addLayer(12, Activation::ELU);
    nn->addLayer(12, Activation::ELU);
    nn->addLayer(1, Activation::SIGMOID);

    // Generate training data
    int data_number_train{250};
    Matrix X_train{Matrix(2,data_number_train)}, Y_train{Matrix(1,data_number_train)};
    generate_data_circle(X_train, Y_train,true);

    nn->train(X_train,Y_train, 1000, 1.1f,true);
}
     
App::~App() { if(nn != nullptr) delete nn; }

void App::update(sf::Time deltaTime) {}

void App::notify(sf::Keyboard::Key key, bool pressed) {
    if(key == sf::Keyboard::Space && pressed) {
        // Generate testing data
        int data_number_test{1000};
        Matrix X_test{Matrix(2,data_number_test)}, Y_test{Matrix(1,data_number_test)};
        generate_data_circle(X_test, Y_test, true);

        // Predictions
        Matrix pred{nn->predict(X_test, Y_test)};
        int good_answer_num{0};
        int res{0};
        for(int i=0; i< Y_test.col(); i++) {
            if(pred.getCoeff(0,i)>=0.5) res=1;
            else res=0;
            if(res == Y_test.getCoeff(0,i)) good_answer_num++;
        }
        std::cout << "Score: " << good_answer_num << "/" << Y_test.col() << std::endl; 
    }
}

void App::draw(sf::RenderTarget& target, sf::RenderStates states) const {
    target.draw(m_data_coord);
    target.draw(m_frontier);
}

// This function generate linearly separable data with 2 class
// X_feature max dim : row=2, col= nb_of_data
// Y_class max dim: row=1, col=nb_of_data
void App::generate_data_linear(Matrix& X_feature, Matrix& Y_class, bool update_graphics) {    
    const float a{-1.4f}, b{1.0f}, c{0.3f};
    const float zoom{500.f};
    sf::Color class_color{sf::Color::Red};
    int nb_of_class0=0;
    if(update_graphics) {
        m_data_coord = sf::VertexArray(sf::Quads,X_feature.col()*4);
        m_frontier = sf::VertexArray(sf::Lines, 2);  // Used to draw frontier with SFML
    }

    // Draw frontier between class 0 and class 1
    m_frontier[0].position = sf::Vector2f((-c/a)*zoom,1.0f*zoom);
    m_frontier[1].position = sf::Vector2f(((-(b+c)/a))*zoom,0.f);
    m_frontier[0].color = sf::Color::White;
    m_frontier[1].color = sf::Color::White;

    // Create random data
    for(int i=0; i<X_feature.col(); i++) {
        for(int j=0; j<X_feature.row(); j++) X_feature.setCoeff(j,i,randomf(0,1));
        
        if(update_graphics) {
            m_data_coord[i*4].position = sf::Vector2f(X_feature.getCoeff(0,i)*zoom-2,(1.0f-X_feature.getCoeff(1,i))*zoom-2);
            m_data_coord[i*4+1].position = sf::Vector2f(X_feature.getCoeff(0,i)*zoom-2,(1.0f-X_feature.getCoeff(1,i))*zoom+2);
            m_data_coord[i*4+2].position = sf::Vector2f(X_feature.getCoeff(0,i)*zoom+2,(1.0f-X_feature.getCoeff(1,i))*zoom+2);
            m_data_coord[i*4+3].position = sf::Vector2f(X_feature.getCoeff(0,i)*zoom+2,(1.0f-X_feature.getCoeff(1,i))*zoom-2);
        }

        if(X_feature.getCoeff(0,i)*a + X_feature.getCoeff(1,i)*b +c >= 0) { // Class 0
            Y_class.setCoeff(0,i,0);
            class_color = sf::Color::Red;
            nb_of_class0++;
        }
        else { // Class 1
            Y_class.setCoeff(0,i,1);
            class_color = sf::Color::Green;
        }

        if(update_graphics) {
            m_data_coord[i*4].color = class_color;
            m_data_coord[i*4+1].color = class_color;
            m_data_coord[i*4+2].color = class_color;
            m_data_coord[i*4+3].color = class_color;
        }
    }
    std::cout << "data created [class 0: " << nb_of_class0 << "(" << ((float)nb_of_class0/(float)X_feature.col())*100.f <<"%)]" << std::endl;
    std::cout << std::setw(23)<< " [class 1: " << X_feature.col() - nb_of_class0 << "(" << ((float)(X_feature.col() - nb_of_class0)/(float)X_feature.col())*100.f <<"%)]" << std::endl;  
}

// This function generate non linearly separable data with 2 class
// X_feature max dim : row=2, col= nb_of_data
// Y_class max dim: row=1, col=nb_of_data
void App::generate_data_circle(Matrix& X_feature, Matrix& Y_class, bool update_graphics) {    
    const float r{0.4f}, x{0.5f}, y{0.5f};
    const float zoom{500.f};
    sf::Color class_color{sf::Color::Red};
    int nb_of_class0=0;
    if(update_graphics) {
        m_data_coord = sf::VertexArray(sf::Quads,X_feature.col()*4);
        m_frontier = sf::VertexArray(sf::Lines, 2);  // Used to draw frontier with SFML
    }

    // Create random data
    for(int i=0; i<X_feature.col(); i++) {
        for(int j=0; j<X_feature.row(); j++) X_feature.setCoeff(j,i,randomf(0,1));
        
        if(update_graphics) {
            m_data_coord[i*4].position = sf::Vector2f(X_feature.getCoeff(0,i)*zoom-2,(1.0f-X_feature.getCoeff(1,i))*zoom-2);
            m_data_coord[i*4+1].position = sf::Vector2f(X_feature.getCoeff(0,i)*zoom-2,(1.0f-X_feature.getCoeff(1,i))*zoom+2);
            m_data_coord[i*4+2].position = sf::Vector2f(X_feature.getCoeff(0,i)*zoom+2,(1.0f-X_feature.getCoeff(1,i))*zoom+2);
            m_data_coord[i*4+3].position = sf::Vector2f(X_feature.getCoeff(0,i)*zoom+2,(1.0f-X_feature.getCoeff(1,i))*zoom-2);
        }

        if((X_feature.getCoeff(0,i)-x)*(X_feature.getCoeff(0,i)-x) + (X_feature.getCoeff(1,i)-y)*(X_feature.getCoeff(1,i)-y) > r*r) { // Class 0
            Y_class.setCoeff(0,i,0);
            class_color = sf::Color::Red;
            nb_of_class0++;
        }
        else { // Class 1
            Y_class.setCoeff(0,i,1);
            class_color = sf::Color::Green;
        }

        if(update_graphics) {
            m_data_coord[i*4].color = class_color;
            m_data_coord[i*4+1].color = class_color;
            m_data_coord[i*4+2].color = class_color;
            m_data_coord[i*4+3].color = class_color;
        }
    }
    std::cout << "data created [class 0: " << nb_of_class0 << "(" << ((float)nb_of_class0/(float)X_feature.col())*100.f <<"%)]" << std::endl;
    std::cout << std::setw(23)<< " [class 1: " << X_feature.col() - nb_of_class0 << "(" << ((float)(X_feature.col() - nb_of_class0)/(float)X_feature.col())*100.f <<"%)]" << std::endl;  
}

// This function generate non linearly separable data with 2 class
// X_feature max dim : row=2, col= nb_of_data
// Y_class max dim: row=1, col=nb_of_data
void App::generate_data_balanced(Matrix& X_feature, Matrix& Y_class, bool update_graphics) {    
    const float r{0.4f}, x{0.5f}, y{0.5f};
    const float zoom{500.f};
    sf::Color class_color{sf::Color::Red};
    int nb_of_class0=0;
    if(update_graphics) {
        m_data_coord = sf::VertexArray(sf::Quads,X_feature.col()*4);
        m_frontier = sf::VertexArray(sf::Lines, 2);  // Used to draw frontier with SFML
    }

    // Create random data
    for(int i=0; i<X_feature.col(); i++) {
        for(int j=0; j<X_feature.row(); j++) X_feature.setCoeff(j,i,randomf(0,1));
        
        if(update_graphics) {
            m_data_coord[i*4].position = sf::Vector2f(X_feature.getCoeff(0,i)*zoom-2,(1.0f-X_feature.getCoeff(1,i))*zoom-2);
            m_data_coord[i*4+1].position = sf::Vector2f(X_feature.getCoeff(0,i)*zoom-2,(1.0f-X_feature.getCoeff(1,i))*zoom+2);
            m_data_coord[i*4+2].position = sf::Vector2f(X_feature.getCoeff(0,i)*zoom+2,(1.0f-X_feature.getCoeff(1,i))*zoom+2);
            m_data_coord[i*4+3].position = sf::Vector2f(X_feature.getCoeff(0,i)*zoom+2,(1.0f-X_feature.getCoeff(1,i))*zoom-2);
        }

        if(X_feature.getCoeff(0,i) < 0.5 && X_feature.getCoeff(1,i) < 0.5 || X_feature.getCoeff(0,i) > 0.5 && X_feature.getCoeff(1,i) > 0.5){ // Class 0
            Y_class.setCoeff(0,i,0);
            class_color = sf::Color::Red;
            nb_of_class0++;
        }
        else { // Class 1
            Y_class.setCoeff(0,i,1);
            class_color = sf::Color::Green;
        }

        if(update_graphics) {
            m_data_coord[i*4].color = class_color;
            m_data_coord[i*4+1].color = class_color;
            m_data_coord[i*4+2].color = class_color;
            m_data_coord[i*4+3].color = class_color;
        }
    }
    std::cout << "data created [class 0: " << nb_of_class0 << "(" << ((float)nb_of_class0/(float)X_feature.col())*100.f <<"%)]" << std::endl;
    std::cout << std::setw(23)<< " [class 1: " << X_feature.col() - nb_of_class0 << "(" << ((float)(X_feature.col() - nb_of_class0)/(float)X_feature.col())*100.f <<"%)]" << std::endl;  
}

// This function generate non linearly separable data with 3 class
// X_feature max dim : row=2, col= nb_of_data
// Y_class max dim: row=3, col=nb_of_data
void App::generate_data_3_class(Matrix& X_feature,Matrix& Y_class, bool update_graphics) {
    const float zoom{500.f};
    sf::Color class_color{sf::Color::Red};
    int nb_of_class0=0;
    int nb_of_class1=0;
    int nb_of_class2=0;
    if(update_graphics) {
        m_data_coord = sf::VertexArray(sf::Quads,X_feature.col()*4);
    }

    // Create random data
    for(int i=0; i<X_feature.col(); i++) {
        for(int j=0; j<X_feature.row(); j++) X_feature.setCoeff(j,i,randomf(0,1));
        
        if(update_graphics) {
            m_data_coord[i*4].position = sf::Vector2f(X_feature.getCoeff(0,i)*zoom-2,(1.0f-X_feature.getCoeff(1,i))*zoom-2);
            m_data_coord[i*4+1].position = sf::Vector2f(X_feature.getCoeff(0,i)*zoom-2,(1.0f-X_feature.getCoeff(1,i))*zoom+2);
            m_data_coord[i*4+2].position = sf::Vector2f(X_feature.getCoeff(0,i)*zoom+2,(1.0f-X_feature.getCoeff(1,i))*zoom+2);
            m_data_coord[i*4+3].position = sf::Vector2f(X_feature.getCoeff(0,i)*zoom+2,(1.0f-X_feature.getCoeff(1,i))*zoom-2);
        }

        if(X_feature.getCoeff(0,i) > 0.5 && X_feature.getCoeff(1,i) > 0.5) { // Class 0
            Y_class.setCoeff(0,i,1);
            Y_class.setCoeff(1,i,0);
            Y_class.setCoeff(2,i,0);
            class_color = sf::Color::Red;
            nb_of_class0++;
        }
        else if (X_feature.getCoeff(0,i) <= 0.5 && X_feature.getCoeff(1,i) > 0.5) { // Class 1
            Y_class.setCoeff(0,i,0);
            Y_class.setCoeff(1,i,1);
            Y_class.setCoeff(2,i,0);
            class_color = sf::Color::Green;
            nb_of_class1++;
        }
        else if (X_feature.getCoeff(1,i) <= 0.5) {
            Y_class.setCoeff(0,i,0);
            Y_class.setCoeff(1,i,0);
            Y_class.setCoeff(2,i,1);
            class_color = sf::Color::Red;
            nb_of_class2++;
        }

        if(update_graphics) {
            m_data_coord[i*4].color = class_color;
            m_data_coord[i*4+1].color = class_color;
            m_data_coord[i*4+2].color = class_color;
            m_data_coord[i*4+3].color = class_color;
        }
    }
    std::cout << "data created [class 0: " << nb_of_class0 << "(" << ((float)nb_of_class0/(float)X_feature.col())*100.f <<"%)]" << std::endl;
    std::cout << std::setw(23)<< " [class 1: " << nb_of_class1 << "(" << ((float)nb_of_class1/(float)X_feature.col())*100.f <<"%)]" << std::endl;  
    std::cout << std::setw(23)<< " [class 2: " << nb_of_class2 << "(" << ((float)nb_of_class2/(float)X_feature.col())*100.f <<"%)]" << std::endl;  
}
