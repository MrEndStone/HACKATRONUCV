<!DOCTYPE html>
<html lang="en">
<head>
    <?php
      $title = "";  
      include '../Components/head.php';
    ?>
    <link rel="stylesheet" href="../Css/alert.css">
    <link rel="stylesheet" href="../Css/app.css">
</head>
<body>
<section class="layout">
    <?php include '../Layouts/topbar.php'; ?>
    <?php 
        if(isset($_GET['page'])) {
            $selected = $_GET['page'];
            if($_GET['page'] == 'newReserve' || $_GET['page'] == 'calendario' || $_GET['page'] == 'pago' || $_GET['page'] == 'verificar') {
                $selected = 'reservas';
            }
        } else {
            $selected = "dashboard";
        }
        include '../Layouts/sidebar.php'; 
    ?>
    <main>
        <?php
            if(!isset($_GET['page']) || $_GET['page'] == "dashboard") {
                include '../Layouts/dashboard.php';
            }
            if(isset($_GET['page']) && $_GET['page'] == "reservas") {
                include '../Layouts/reservas.php';
            }
            if(isset($_GET['page']) && $_GET['page'] == "comprobantes") {
                include '../Layouts/comprobantes.php';
            }
            if(isset($_GET['page']) && $_GET['page'] == "perfil") {
                include '../Layouts/perfil.php';
            }
            if(isset($_GET['page']) && $_GET['page'] == "nuevaReserva") {
                include '../Layouts/newReserve.php';
            }
            if(isset($_GET['page']) && $_GET['page'] == "calendario") {
                include '../Layouts/calendar.php';
            }
            if(isset($_GET['page']) && $_GET['page'] == "pago") {
                include '../Layouts/payment.php';
            }
            if(isset($_GET['page']) && $_GET['page'] == "verificar") {
                include '../Layouts/verification.php';
            }
        ?>
    </main>

    <div class="alert block" id="appAlert">
        <div>
            <i class='bx bxs-x-circle' id="closeLoginAlert"></i>
            <span></span>
        </div>  
    </div>
    <script src="../Scripts/app.js" type="module"></script>
</section>
</body>
</html>