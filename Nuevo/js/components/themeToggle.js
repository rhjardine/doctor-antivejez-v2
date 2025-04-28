// js/components/themeToggle.js

function createThemeToggleHTML() {
    // Return the HTML for the theme toggle button
    return `
        <div class="theme-toggle" id="theme-toggle" title="Cambiar Tema">
            <i class="fas fa-moon"></i>
        </div>
    `;
}

function initializeThemeToggle() {
    const themeToggleBtn = document.getElementById('theme-toggle');
    if (!themeToggleBtn) {
        console.warn("Theme toggle button not found.");
        return;
    }

    const body = document.body;
    const icon = themeToggleBtn.querySelector('i');

    const applyTheme = (isDark) => {
        body.classList.toggle('dark-mode', isDark);
        if (icon) {
            icon.classList.toggle('fa-sun', isDark);
            icon.classList.toggle('fa-moon', !isDark);
        }
        // Optional: Add event dispatch if other components need to know theme changed
        // document.dispatchEvent(new CustomEvent('themeChanged', { detail: { isDarkMode: isDark } }));
    };

    // Check initial theme preference
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const savedTheme = localStorage.getItem('theme');
    const initialThemeIsDark = savedTheme === 'dark' || (!savedTheme && prefersDark);
    applyTheme(initialThemeIsDark); // Apply initial theme

    // Add click listener
    themeToggleBtn.addEventListener('click', () => {
        const isDarkMode = body.classList.toggle('dark-mode');
        applyTheme(isDarkMode); // Update icon
        localStorage.setItem('theme', isDarkMode ? 'dark' : 'light');
    });

    console.log("Theme toggle initialized.");
}