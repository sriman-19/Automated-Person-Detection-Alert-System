// Video Processing specific JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Configuration for video processing
    const videoConfig = {
        processingDelay: 500, // ms between frames
        confidenceThreshold: 0.7, // default confidence threshold
        maxDuration: 1800 // max video duration in seconds (30 minutes)
    };

    // Initialize video processing components if on the video processing page
    if (document.getElementById('videoForm')) {
        initVideoProcessing();
    }

    function initVideoProcessing() {
        // Get form elements
        const videoForm = document.getElementById('videoForm');
        const videoInput = document.getElementById('video');
        const confidenceSlider = document.getElementById('confidence');
        const highAccuracyCheckbox = document.getElementById('highAccuracy');
        const processButton = document.getElementById('processButton');

        // Add validation for video files
        videoInput.addEventListener('change', validateVideoFile);

        // Update confidence threshold when slider changes
        confidenceSlider.addEventListener('input', function() {
            const confidenceValue = parseInt(this.value) / 100;
            videoConfig.confidenceThreshold = confidenceValue;
            document.getElementById('confidenceValue').textContent = this.value + '%';
        });

        // Update processing mode when checkbox changes
        highAccuracyCheckbox.addEventListener('change', function() {
            if (this.checked) {
                videoConfig.processingDelay = 200; // Process more frames
            } else {
                videoConfig.processingDelay = 500; // Process fewer frames
            }
        });
    }

    function validateVideoFile(event) {
        const file = event.target.files[0];
        if (!file) return;

        // Check file type
        const fileType = file.type;
        const validTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-msvideo'];

        if (!validTypes.includes(fileType)) {
            alert('Please select a valid video file (MP4, AVI, MOV)');
            this.value = ''; // Clear the file input
            return false;
        }

        // Check file size (100MB limit)
        const maxSize = 20 * 1024 * 1024 * 1024; // 100MB in bytes
        if (file.size > maxSize) {
            alert('File size exceeds the 20GB limit. Please select a smaller file.');
            this.value = ''; // Clear the file input
            return false;
        }

        return true;
    }
});

// Function to display processing progress
function updateProcessingProgress(percentage) {
    const progressBar = document.getElementById('progressBar');
    if (progressBar) {
        progressBar.style.width = percentage + '%';
        progressBar.setAttribute('aria-valuenow', percentage);
    }
}

// Function to show processing spinner
function showProcessingSpinner() {
    const spinner = document.getElementById('processingSpinner');
    const processButton = document.getElementById('processButton');

    if (spinner && processButton) {
        spinner.classList.remove('d-none');
        processButton.disabled = true;
    }
}

// Function to hide processing spinner
function hideProcessingSpinner() {
    const spinner = document.getElementById('processingSpinner');
    const processButton = document.getElementById('processButton');

    if (spinner && processButton) {
        spinner.classList.add('d-none');
        processButton.disabled = false;
    }
}