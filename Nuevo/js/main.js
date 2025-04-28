// js/main.js

document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM fully loaded. Initializing main listeners...");

    // Initialize Theme Toggle Logic (which includes listeners)
    if (typeof initializeThemeToggle === 'function') {
        initializeThemeToggle();
    } else {
        console.error("initializeThemeToggle function not found!");
    }

    // Initialize Sidebar Toggle Logic
    const sidebarEl = document.getElementById('sidebar'); // Find the rendered sidebar
    const mainContentEl = document.getElementById('main-content-placeholder'); // Target the placeholder
    const sidebarToggleTrigger = document.getElementById('toggle-sidebar'); // Trigger is the logo div

    if (sidebarEl && sidebarToggleTrigger && mainContentEl) {
        const adjustMargin = () => {
            // Only adjust margin if not on mobile layout
            if (window.innerWidth > 480) {
                mainContentEl.style.marginLeft = sidebarEl.classList.contains('collapsed') ? '64px' : '220px';
            } else {
                 // Ensure margin is removed on mobile regardless of collapse state
                mainContentEl.style.marginLeft = '0';
            }
        };

        sidebarToggleTrigger.addEventListener('click', (e) => {
            e.preventDefault(); // Prevent link behavior if logo is link
            sidebarEl.classList.toggle('collapsed');
            adjustMargin(); // Adjust margin on toggle
        });

        // Initial margin adjustment and on resize
        adjustMargin();
        window.addEventListener('resize', adjustMargin);
        console.log("Sidebar toggle listener initialized.");

    } else {
        console.warn("Sidebar components for toggle not found (might be expected on pages without sidebar).");
    }

    // Initialize other global listeners or functionality here...

    console.log("Main listeners initialized.");

}); // End DOMContentLoaded

// Make helper functions globally available if not using modules (or export/import if using modules)
// window.showToast = showToast; // Example if helpers.js isn't treated as a module