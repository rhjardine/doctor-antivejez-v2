// js/modules/helpers.js

function showToast(title, message, type = 'info') {
    const toastContainer = document.getElementById('toast-container');
    if(!toastContainer) {
        console.warn('Toast container not found!');
        return;
    }

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    const iconClass = type === 'success' ? 'fas fa-check-circle' :
                      type === 'warning' ? 'fas fa-exclamation-triangle' :
                      type === 'error'   ? 'fas fa-times-circle' :
                                           'fas fa-info-circle'; // Default info
    toast.innerHTML = `
        <div class="toast-icon ${type}"><i class="${iconClass}"></i></div>
        <div class="toast-content">
            <div class="toast-title">${title}</div>
            <div class="toast-message">${message}</div>
        </div>
        <button class="toast-close"><i class="fas fa-times"></i></button>
    `;
    toastContainer.appendChild(toast);

    const closeButton = toast.querySelector('.toast-close');
    closeButton?.addEventListener('click', () => {
        toast.classList.remove('show');
        setTimeout(() => { if (toast.parentNode) toast.remove(); }, 300);
    });

    setTimeout(() => toast.classList.add('show'), 10);

    setTimeout(() => {
        if (toast.parentNode === toastContainer) {
            toast.classList.remove('show');
            setTimeout(() => {
                 if (toast.parentNode === toastContainer) toast.remove();
             }, 300);
        }
    }, 5000);
}

function formatFileSize(bytes) {
    const size = Number(bytes);
    if (isNaN(size) || size === 0) return '0 B';
    const i = Math.floor(Math.log(size) / Math.log(1024));
    return parseFloat((size / Math.pow(1024, i)).toFixed(1)) + ' ' + ['B', 'KB', 'MB', 'GB'][i];
}

function getFileIcon(file) {
    if (!file || !file.type) return 'fa-file';
    const type = file.type.toLowerCase();
    if (type.includes('pdf')) return 'fa-file-pdf';
    if (type.includes('image')) return 'fa-file-image';
    if (type.includes('word')) return 'fa-file-word';
    if (type.includes('excel') || type.includes('spreadsheet')) return 'fa-file-excel';
    // Add more types as needed
    return 'fa-file-alt'; // Generic file
}

// Add other helper functions as needed...