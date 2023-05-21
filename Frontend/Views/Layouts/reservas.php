<link rel="stylesheet" href="../Css/reservas.css">

<section class="reservas">
    <div class="row--right">
        <?php $nameButton = "newReserva"; $textButton = "Nueva Reserva";  $isTransparent = false;   include '../Components/button.php'; ?>
    </div>
    <table class="data">
        <thead>
            <tr>
                <th>Fecha</th>
                <th>Estado</th>
                <th>Servicio</th>
            </tr>
        </thead>
        <tbody>
        </tbody>
    </table>
</section>

<div class="alert block" id="servicesDetails">
    <div>
        <i class='bx bxs-x-circle' id="closeServicesDetails"></i>
        <table class="data">
            <thead>
                <tr>
                    <th>Servicio</th>
                    <th>Categoría</th>
                    <th>Duración</th>
                    <th>Precio</th>
                </tr>
            </thead>
            <tbody>
            </tbody>
        </table>
    </div>  
</div>

<script src="../Scripts/reservas.js" type="module"></script>