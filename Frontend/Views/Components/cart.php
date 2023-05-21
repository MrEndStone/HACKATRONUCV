<link rel="stylesheet" href="../Css/cart.css">

<section class="cart--bg">
    <div class="cart" id="<?= $idCart ?>">
        <div class="cart__header">
            <span class="cart__username"></span>
            <i class='bx bxs-x-circle' id="closeCart"></i>
        </div>
    
        <div class="cart__details">
            <div class="cart__details--header">
                <div>Servicio</div>
                <div>Precio</div>
                <div>Duración</div>
            </div>
    
            <div class="cart__details--body">
            </div>
    
            <div class="line" style="width:80%"></div>

            <div class="cart__details--foot">
                <div class="cart__key--foot">Total</div>
                <div class="cart__price--foot"></div>
                <div class="cart__duration--foot"></div>
            </div>
        </div>

        <a href="http://localhost/Projects/RapiReserva/Views/Pages/app.php?page=calendario" class="cart__btn">Pagar Reserva</a>
    </div>
</section>