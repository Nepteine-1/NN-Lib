#ifndef APP_HPP
#define APP_HPP

#include <vector>
#include <iostream>
#include <iomanip>

#include <SFML/Graphics.hpp>
#include <Engine/EventHandler.hpp>
#include <NeuralNetwork/Matrix.hpp>
#include <NeuralNetwork/NeuralNetwork.hpp>
#include <exception>

class App : public sf::Drawable, public KeyBoardObserver {
    public:
        App();
        virtual ~App();
        void update(sf::Time deltaTime);
        virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const;
        virtual void notify(sf::Keyboard::Key key, bool pressed);

    private:
        void generate_data_linear(Matrix& X_feature,Matrix& Y_class, bool update_graphics=false);
        void generate_data_circle(Matrix& X_feature,Matrix& Y_class, bool update_graphics=false);
        void generate_data_balanced(Matrix& X_feature,Matrix& Y_class, bool update_graphics=false);
        void generate_data_3_class(Matrix& X_feature,Matrix& Y_class, bool update_graphics=false);
    
    private:
        sf::VertexArray m_data_coord; 
        sf::VertexArray m_frontier;
        NeuralNetwork* nn = nullptr;
};

#endif
