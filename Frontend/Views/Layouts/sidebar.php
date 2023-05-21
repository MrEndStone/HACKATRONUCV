<link rel="stylesheet" href="../Css/sidebar.css">

<aside class="aside">
    <div class="aside__logo">
        <i class='bx bxs-crown'></i>
      <span>JQueens</span>
    </div>
    <div class="aside__menu">
        <?php $icon = "bx bxs-home-smile";  $text = "Dashboard";      $url = "../Pages/app.php?page=dashboard"; $active = $selected == "dashboard" ? "selected" : "";  include '../Components/sidebarButton.php'; ?>
        <?php $icon = "bx bxs-calendar";    $text = "Reservas";       $url = "../Pages/app.php?page=reservas"; $active = $selected == "reservas" ? "selected" : "";  include '../Components/sidebarButton.php'; ?>
        <?php $icon = "bx bxs-notepad";     $text = "Comprobantes";   $url = "../Pages/app.php?page=comprobantes"; $active = $selected == "comprobantes" ? "selected" : "";  include '../Components/sidebarButton.php'; ?>
        <?php $icon = "bx bxs-user";        $text = "Perfil";         $url = "../Pages/app.php?page=perfil"; $active = $selected == "perfil" ? "selected" : "";  include '../Components/sidebarButton.php'; ?>
        <?php $icon = "bx bx-exit";         $text = "Cerrar Sesión";  $url = "../Pages/app.php?page=logout";  $active = "logout"; include '../Components/sidebarButton.php'; ?>
    </div>
  </aside>

<script src="../Scripts/sidebar.js" type="module"></script>