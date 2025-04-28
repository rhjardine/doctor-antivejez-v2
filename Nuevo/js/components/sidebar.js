// js/components/sidebar.js

function createSidebarHTML(activePageFilename = 'dashboard_advanced_gen.html') { // Default active page
    // Define menu items
    const menuItems = [
        { href: "dashboard_advanced_gen.html", icon: "fa-chart-line", text: "Dashboard" },
        { href: "historias_tabs_bio.html", icon: "fa-book-medical", text: "Historias" },
        { href: "profesionales.html", icon: "fa-user-md", text: "Profesionales" },
        { href: "#ai-assistant", icon: "fa-robot", text: "Agente IA" }, // Use fragment ID or specific page
        { href: "#reports", icon: "fa-file-medical-alt", text: "Reportes" },
        { href: "#settings", icon: "fa-cog", text: "Ajustes" },
        { href: "#logout", icon: "fa-sign-out-alt", text: "Salir" }
    ];

    // Generate the HTML for menu items
    const menuItemsHTML = menuItems.map(item => {
        // Basic check: compare filename part of href with activePageFilename
        const itemFilename = item.href.includes('/') ? item.href.split('/').pop() : item.href;
        const isActive = itemFilename === activePageFilename ||
                         (item.href.startsWith('#') && activePageFilename === 'dashboard_advanced_gen.html' && item.href === '#ai-assistant'); // Example: Make AI active on dashboard default

        return `
            <li class="nav-item ${isActive ? 'active' : ''}">
                <a href="${item.href}">
                    <i class="fas ${item.icon}"></i>
                    <span>${item.text}</span>
                </a>
            </li>
        `;
    }).join('');

    // Return the complete sidebar HTML structure
    return `
        <div class="sidebar" id="sidebar">
            <div class="sidebar-header">
                <a href="dashboard_advanced_gen.html" class="logo" id="toggle-sidebar" aria-label="Volver al Dashboard">
                    <img src="assets/logo.png" alt="Logo Doctor Antivejez" class="logo-img">
                </a>
            </div>
            <ul class="nav-menu">
                ${menuItemsHTML}
            </ul>
            ${createUserProfileHTML()}
        </div>
    `;
}

// You might place this in a separate userProfile.js or keep it here
function createUserProfileHTML(name = "Dr. García", role = "Especialista") {
     return `
         <div class="user-profile">
            <div class="user-avatar"><i class="fas fa-user-md"></i></div>
            <div class="user-info">
                <div class="user-name">${name}</div>
                <div class="user-role">${role}</div>
            </div>
        </div>
     `;
 }