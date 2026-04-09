pipeline {
    agent any

    environment {
        APP_NAME = 'inference-api'
        APP_PORT = '8000'
        IMAGE_TAG = "${APP_NAME}:${env.BUILD_NUMBER}"
        IMAGE_LATEST = "${APP_NAME}:latest"
    }

    options {
        timestamps()
        disableConcurrentBuilds()
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Sanity') {
            steps {
                sh '''
                    python3 -m venv .venv
                    . .venv/bin/activate
                    pip install --upgrade pip
                    pip install pre-commit black isort flake8 pytest
                    pre-commit run --all-files
                '''
            }
        }

        stage('Unit Smoke Test') {
            steps {
                sh '''
                    . .venv/bin/activate
                    pytest -q tests/test_smoke.py
                '''
            }
        }

        stage('Build Image') {
            steps {
                sh '''
                    docker build -t ${IMAGE_TAG} -t ${IMAGE_LATEST} .
                '''
            }
        }

        stage('Runtime Smoke Test') {
            steps {
                sh '''
                    docker rm -f ${APP_NAME}-test || true
                    docker run -d --name ${APP_NAME}-test -p 18000:8000 ${IMAGE_TAG}
                    sleep 10
                    curl --fail http://127.0.0.1:18000/health
                    docker rm -f ${APP_NAME}-test
                '''
            }
        }

        stage('Deploy') {
            steps {
                sh '''
                    docker rm -f ${APP_NAME} || true
                    docker run -d \
                      --name ${APP_NAME} \
                      --restart unless-stopped \
                      -p ${APP_PORT}:8000 \
                      ${IMAGE_LATEST}
                '''
            }
        }

        stage('Post Deploy Check') {
            steps {
                sh '''
                    sleep 5
                    curl --fail http://127.0.0.1:${APP_PORT}/health
                '''
            }
        }
    }

    post {
        success {
            echo 'Deployment succeeded.'
        }
        failure {
            echo 'Pipeline failed.'
            sh 'docker ps -a || true'
            sh 'docker logs ${APP_NAME}-test || true'
            sh 'docker logs ${APP_NAME} || true'
        }
    }
}
