<link rel="stylesheet" href="../Css/topbar.css">

<header class="header topbar">
    <div class="header--left">
      <h2 class="header__title">
        <?php
          if(isset($_GET['page']) && $_GET['page'] == 'nuevaReserva') {
            echo 'Nueva Reserva';
          } else if (isset($_GET['page']) && $_GET['page'] == 'pago') {
            echo 'Medio de pago';
          } else if (isset($_GET['page']) && $_GET['page'] == 'verificar') {
            echo 'Detalle de compra';
          } else {
            echo isset($_GET['page']) ? $_GET['page'] : 'dashboard';
          }
        ?>
      </h2>
    </div>
    <div class="header--right">
      <span class="header__username ">Username</span>
      <?php $icon = "bx bxs-bell"; $id = ""; $className = 'notificacion';   include '../Components/icon.php'; ?>
      <?php $icon = null; $imageUrl = "../Assets/user-default.png"; $id = ""; $className = 'header__photo';  include '../Components/icon.php'; ?>
    </div>
</header>

<script src="../Scripts/topbar.js" type="module"></script>
