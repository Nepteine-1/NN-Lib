pipeline {
    agent any
    
    stages {
        stage('Build') {
            steps {
                echo 'Building..'
                sh '''
                export TERM=xterm
                
                sh install.sh static release
                sh install.sh example
                ./example/Example
                '''
                
            }
        }
        stage('Test') {
            steps {
                echo 'Testing..'
                sh '''
                echo "doing test stuff.."
                '''
                
            }
        }
        stage('Deliver') {
            steps {
                echo 'Delivering..'
                sh '''
                echo "doing deliver stuff.."
                '''
                
            }
        }
    }
}
