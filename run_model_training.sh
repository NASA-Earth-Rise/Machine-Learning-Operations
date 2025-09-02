#!/bin/bash
# /servir_apps/ml_ops_project/run_model_training.sh

# Configuration
PROJECT_DIR="/servir_apps/ml_ops_project"
LOG_DIR="${PROJECT_DIR}/logs"
LOCK_FILE="${PROJECT_DIR}/.model_training.lock"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="${LOG_DIR}/model_training_${TIMESTAMP}.log"
MAX_RUNTIME=7200  # Maximum runtime in seconds (2 hours)
CONDA_ENV="ml_ops_project"

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

# Check if previous job is still running
if [ -f "${LOCK_FILE}" ]; then
    LOCK_PID=$(cat "${LOCK_FILE}")
    if ps -p "${LOCK_PID}" > /dev/null 2>&1; then
        log_message "Previous training job (PID: ${LOCK_PID}) still running. Skipping."
        exit 1
    else
        log_message "Stale lock file found. Previous job likely crashed. Continuing..."
        rm -f "${LOCK_FILE}"
    fi
fi

# Create lock file with current PID
echo $$ > "${LOCK_FILE}"

# Ensure lock file is removed when script exits
trap 'rm -f "${LOCK_FILE}"; log_message "Training pipeline completed or interrupted."' EXIT

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Function to run a Python script with timeout and error checking
run_script() {
    local script_name="$1"
    local script_path="${PROJECT_DIR}/${script_name}"

    log_message "Starting ${script_name}..."

    # Activate conda environment and run script
    if timeout "${MAX_RUNTIME}" bash -c "
        source activate \"${CONDA_ENV}\" 2>/dev/null || conda activate \"${CONDA_ENV}\"
        cd \"${PROJECT_DIR}\"
        python \"${script_path}\"
    " 2>&1 | tee -a "${LOG_FILE}"; then
        log_message "✅ ${script_name} completed successfully"
        return 0
    else
        local exit_code=$?
        if [ ${exit_code} -eq 124 ] || [ ${exit_code} -eq 137 ]; then
            log_message "❌ ${script_name} timed out after $((MAX_RUNTIME / 60)) minutes"
        else
            log_message "❌ ${script_name} failed with exit code ${exit_code}"
        fi
        return ${exit_code}
    fi
}

# Start the pipeline
log_message "🚀 Starting NLU model training pipeline..."
log_message "Using conda environment: ${CONDA_ENV}"
log_message "Working directory: ${PROJECT_DIR}"

# Step 1: Generate ML Data
if ! run_script "generate_ml_data.py"; then
    log_message "💥 Pipeline failed at data generation step"
    exit 1
fi

# Step 2: Data Preprocessing
if ! run_script "data_preprocessing.py"; then
    log_message "💥 Pipeline failed at data preprocessing step"
    exit 1
fi

# Step 3: Train NLU Model
if ! run_script "train_nlu_model.py"; then
    log_message "💥 Pipeline failed at model training step"
    exit 1
fi

# Pipeline completed successfully
log_message "🎉 Complete NLU model training pipeline finished successfully!"

# Rotate logs (keep last 10)
find "${LOG_DIR}" -name "model_training_*.log" -type f | sort -r | tail -n +11 | xargs rm -f

# Optional: Restart the nesis_project service to pick up the new model
# Uncomment if you want automatic service restart
# log_message "🔄 Restarting nesis_project service to load new model..."
# sudo systemctl restart your-nesis-service-name

log_message "📊 Pipeline Summary:"
log_message "   - Data Generation: ✅"
log_message "   - Data Preprocessing: ✅"
log_message "   - Model Training: ✅"
log_message "   - Log file: ${LOG_FILE}"

exit 0