pipeline {
    agent any
    
    triggers {
    	pollSCM('* * * * *')
    }
    
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

    post {
        success {
            echo 'Tout s\'est bien passé !'
        }
        failure {
            echo 'Une erreur est survenue.'
        }
    }
}
