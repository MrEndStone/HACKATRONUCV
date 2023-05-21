<link rel="stylesheet" href="../Css/newReserve.css">
<link rel="stylesheet" href="../Css/button.css">

<section class="catalogo">
    <div class="catalogo--header">
        <a href="http://localhost/Projects/RapiReserva/Views/Pages/app.php?page=reservas" class="btn">Volver</a>
        <?php $icon = "bx bxs-cart"; $id = "cartIcon"; $imageUrl = null; $className = 'notificacion';   include '../Components/icon.php'; ?>
    </div>

    <h2 class="catalogo__title">Catálogo</h2>
    <div class="line"></div>

    <div class="catalogo--filters" id="catalogoFilters">
    </div>

    <div class="services" id="catalogoServices">
    </div>
    
    <?php $idButtonCart = "btnPayServices"; $textButtonCart = "Pagar Reserva";  $idCart = "idCartDetails"; $urlCartButton = "#"; include '../Components/cart.php'; ?>
</section>

<script src="../Scripts/newReserve.js" type="module"></script>