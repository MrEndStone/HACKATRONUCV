<link rel="stylesheet" href="../Css/dashboard.css">

<section class="dashboard">
    <div class="row">
        <p id="reservationDays"></p>
        <?php $nameButton = "newReserva"; $textButton = "Reservar";  $isTransparent = false;   include '../Components/button.php'; ?>
    </div>

    <section>
        <h2>Recomendaciones</h2>
        <div class="services" id="services">
            <article class="service">
                <div class="service__image">
                    <img src="../Assets/FotoServices/alisado-japones.jpg" alt="">
                </div>
                <div class="service__info">
                    <h3 class="service__name">Alisado Japonés</h3>
                    <?php $nameButton = "newReserva"; $textButton = "Lo quiero";  $isTransparent = false;   include '../Components/button.php'; ?>
                </div>
            </article>
            <article class="service">
                <div class="service__image">
                    <img src="../Assets/FotoServices/balayage.jpg" alt="">
                </div>
                <div class="service__info">
                    <h3 class="service__name">Balayage</h3>
                    <?php $nameButton = "newReserva"; $textButton = "Lo quiero";  $isTransparent = false;   include '../Components/button.php'; ?>
                </div>
            </article>
            <article class="service">
                <div class="service__image">
                    <img src="../Assets/FotoServices/corte-dama.jpg" alt="">
                </div>
                <div class="service__info">
                    <h3 class="service__name">Corte de dama</h3>
                    <?php $nameButton = "newReserva"; $textButton = "Lo quiero";  $isTransparent = false;   include '../Components/button.php'; ?>
                </div>
            </article>
        </div>
    </section>

    <section>
        <h2>Promociones</h2>
        <div class="promotions">
            <?php include '../Components/slider.php'; ?>
        </div>
    </section>
</section>

<script src="../Scripts/dashboard.js" type="module"></script>