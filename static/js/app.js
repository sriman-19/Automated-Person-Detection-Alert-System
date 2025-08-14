// Main application JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Add active class to current nav item
    const currentLocation = window.location.pathname;
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
    navLinks.forEach(link => {
        if (link.getAttribute('href') === currentLocation) {
            link.classList.add('active');
        }
    });

    // Auto-dismiss alerts after 5 seconds
    const alertList = document.querySelectorAll('.alert:not(.alert-warning):not(.alert-danger)');
    alertList.forEach(function(alert) {
        setTimeout(function() {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    });

    // Handle file input change display
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        input.addEventListener('change', function(e) {
            const fileName = this.files[0]?.name;
            const fileLabel = this.nextElementSibling;
            if (fileLabel && fileName) {
                // Update the label with the file name if it's a child of input-group
                const displayName = fileName.length > 20 ? fileName.substring(0, 17) + '...' : fileName;
                if (this.parentElement.classList.contains('input-group')) {
                    const customText = fileLabel.querySelector('.custom-file-label');
                    if (customText) {
                        customText.textContent = displayName;
                    }
                }
            }
        });
    });
});