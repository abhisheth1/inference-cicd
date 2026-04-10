pipeline {
    agent any

    environment {
        IMAGE_NAME = "ct-inference-api"
        CONTAINER_NAME = "ct-inference-api-prod"
        TEST_CONTAINER_NAME = "ct-inference-api-test"
        APP_PORT = "8000"
        TEST_PORT = "18000"
        VENV_DIR = ".venv"
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

        stage('Python Setup') {
            steps {
                sh '''
                    python3 -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate
                    python -m pip install --upgrade pip
                    pip install -r requirements.txt
                    pip install pre-commit pytest
                '''
            }
        }

        stage('Sanitation / Lint / Smoke') {
            steps {
                sh '''
                    . ${VENV_DIR}/bin/activate
                    pre-commit run --all-files
                '''
            }
        }

        stage('Tests') {
            steps {
                sh '''
                    . ${VENV_DIR}/bin/activate
                    pytest -q
                '''
            }
        }

        stage('Docker Build') {
            steps {
                sh '''
                    docker build \
                      -t ${IMAGE_NAME}:${BUILD_NUMBER} \
                      -t ${IMAGE_NAME}:latest .
                '''
            }
        }

        stage('Container Runtime Validation') {
            steps {
                sh '''
                    docker rm -f ${TEST_CONTAINER_NAME} || true

                    docker run -d \
                      --name ${TEST_CONTAINER_NAME} \
                      -e MODEL_OUT_DIR=/app/outputs_candidate_generation \
                      -p ${TEST_PORT}:8000 \
                      ${IMAGE_NAME}:${BUILD_NUMBER}

                    sleep 10

                    curl --fail http://127.0.0.1:${TEST_PORT}/
                    curl --fail http://127.0.0.1:${TEST_PORT}/health > health.json

                    python3 - <<'PY'
                    import json
                    with open("health.json", "r", encoding="utf-8") as f:
                        data = json.load(f)
                    assert data["status"] == "ok", data
                    assert data["generator_weights_exists"] is True, data
                    assert data["classifier_weights_exists"] is True, data
                    assert data["model_ready"] is True, data
                    PY

                    docker rm -f ${TEST_CONTAINER_NAME}
                    rm -f health.json
                '''
            }
        }

        stage('Deploy') {
            steps {
                sh '''
                    docker rm -f ${CONTAINER_NAME} || true

                    docker run -d \
                      --name ${CONTAINER_NAME} \
                      --restart unless-stopped \
                      -e MODEL_OUT_DIR=/app/outputs_candidate_generation \
                      -p ${APP_PORT}:8000 \
                      ${IMAGE_NAME}:${BUILD_NUMBER}
                '''
            }
        }
    }

    post {
        success {
            echo 'Build passed. Lint, smoke test, runtime validation, and deployment all succeeded.'
        }

        failure {
            echo 'Build failed. Deployment was blocked.'
        }

        always {
            sh '''
                docker rm -f ${TEST_CONTAINER_NAME} || true
                rm -f health.json || true
                docker image prune -f || true
            '''
        }
    }
}
