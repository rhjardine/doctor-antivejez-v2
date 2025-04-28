// js/layout.js

function renderLayout() {
    console.log("Rendering layout components...");
    const sidebarPlaceholder = document.getElementById('sidebar-placeholder');
    const themeTogglePlaceholder = document.getElementById('theme-toggle-placeholder');
    const mainContentPlaceholder = document.getElementById('main-content-placeholder'); // Ensure this exists

    if (sidebarPlaceholder && typeof createSidebarHTML === 'function') {
         // Determine active page based on filename
         const currentPage = window.location.pathname.split('/').pop() || 'index.html'; // Assuming index.html is dashboard now
         sidebarPlaceholder.innerHTML = createSidebarHTML(currentPage);
         console.log("Sidebar rendered.");
     } else {
          console.error("Sidebar placeholder or createSidebarHTML function not found!");
      }

     if (themeTogglePlaceholder && typeof createThemeToggleHTML === 'function') {
         themeTogglePlaceholder.innerHTML = createThemeToggleHTML();
         console.log("Theme toggle rendered.");
     } else {
          // Optional: Log warning if placeholder exists but function doesn't
          // console.warn("Theme toggle placeholder or createThemeToggleHTML function not found.");
      }

     // Adjust main content margin after sidebar render (might need slight delay or better timing)
     // This is better handled in main.js after DOM is fully ready
      /*
      if(mainContentPlaceholder && sidebarPlaceholder && sidebarPlaceholder.firstChild) {
          const sidebarElement = sidebarPlaceholder.firstChild; // Assumes the first child is the .sidebar div
          if (window.innerWidth > 480) {
             mainContentPlaceholder.style.marginLeft = sidebarElement.classList.contains('collapsed') ? '64px' : '220px';
          } else {
             mainContentPlaceholder.style.marginLeft = '0';
          }
      }
      */

     console.log("Layout rendering complete.");
}

// Initial call to render the layout as soon as this script runs
// Note: This assumes component functions are already defined (loaded before this script)
renderLayout();