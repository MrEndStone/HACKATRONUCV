const sidebarButtons = document.querySelectorAll(".btn--sidebar");

sidebarButtons.forEach(button => {
  button.addEventListener("click", e => {
    if(e.target.classList.contains("logout") || e.target.parentElement.classList.contains("logout")) {
      e.preventDefault();
      localStorage.clear();
      window.location.replace("http://localhost/Projects/RapiReserva/Views/Pages/login.php");
    }

    sidebarButtons.forEach(btn => btn.classList.remove("selected"));
    button.classList.add("selected");
  }); 
})