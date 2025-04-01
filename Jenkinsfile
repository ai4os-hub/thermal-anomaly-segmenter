@Library(['github.com/indigo-dc/jenkins-pipeline-library@release/2.1.1']) _

def projectConfig

pipeline {
    agent {
        label 'docker'
    }

    stages {
        stage('SQA Model api testing') {
            steps {
                script {
                    projectConfig = pipelineConfig()
                    buildStages(projectConfig)
                }
            }
            post {
                cleanup {
                    cleanWs()
                }
            }
        }
    }
}
