



pipeline {
    agent none 
    options {
        parallelsAlwaysFailFast()
    }
    environment{
        image = "miopen"
    }
    stages{
        // Run all static analysis tests
        stage("Static checks"){
            parallel{
                stage('Clang Tidy') {
                    agent{  label miopentest }
                    steps{
			sh `echo "Hello World."`
                    }
		}
            }
        }
    }    
}

