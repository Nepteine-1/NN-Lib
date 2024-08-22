#include <Engine/Engine.hpp>

Engine::Engine(): m_window(sf::VideoMode(800,600), "Application SFML"), m_eventHandler(EventHandler::getEventHandler())
{
    // Clear the screen
    m_window.clear(sf::Color::Black);
    m_window.display();

    srand(time(NULL));

    m_app = new App();

    m_timePerFrame = sf::seconds(1.f/120.f);
    m_window.setFramerateLimit(120);
    m_window.setKeyRepeatEnabled(false);

    EventHandler::getEventHandler()->addKeyBoardObserver(this);
    EventHandler::getEventHandler()->addEventObserver(this);
}

void Engine::run()
{
    m_timer.add("Clock1");
    sf::Time timeSinceLastUpdate = m_timer.restart("Clock1");

    while (m_window.isOpen())
    {
        m_eventHandler->processEvents(m_window);
        timeSinceLastUpdate += m_timer.restart("Clock1");
        while (timeSinceLastUpdate > m_timePerFrame)
        {
            m_eventHandler->processEvents(m_window); 
            timeSinceLastUpdate -= m_timePerFrame;
            update(m_timePerFrame);
        }
        render();
    }
}

void Engine::notify(sf::Keyboard::Key key, bool pressed) {
    if(key == sf::Keyboard::Escape) { m_window.close(); }
}

void Engine::notify(sf::Event m_event) {
    if(m_event.type == sf::Event::Closed) {
        m_window.close();   
    }   
    else if (m_event.type == sf::Event::Resized)
    {
        // update the window size
        sf::FloatRect visibleArea(0.f, 0.f, m_event.size.width, m_event.size.height);
        m_window.setView(sf::View(visibleArea));
    }
}

void Engine::update(sf::Time deltaTime)
{
    m_app->update(deltaTime);
}

void Engine::render(void)
{
    m_window.clear();
    m_window.draw(*m_app);
    m_window.display();
}

Engine::~Engine(){
    delete m_app;
}