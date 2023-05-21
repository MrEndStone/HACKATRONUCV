<!DOCTYPE html>
<html lang="en">
<head>
    <?php
      $title = "";  
      include '../Components/head.php';
    ?>
    <link rel="stylesheet" href="../Css/alert.css">
    <link rel="stylesheet" href="../Css/login.css">
</head>
<body>
  <div class="container">
    <div class="container--left">
      <div class="panel" id="panel--sign-up">
        <div class="content">
          <h3>¿ Eres nuevo aquí ?</h3>
          <p>Crear una cuenta es rápido y fácil.</p>
          <p>Para iniciar ¡haz clic acá!</p>
          <?php $nameButton = "sign-up-btn";  $textButton = "Crear cuenta";  $isTransparent = true; include '../Components/button.php'; ?>
        </div>
        <div class="image image--left"></div>
      </div>

      <form action="../../Controllers/loginController.php" method="POST" id="form--sign-in" class="form block">
        <h2 class="title">Registrarte</h2>
        <?php $isDisabled = false; $name = "usernameSignup"; $icon = "bx bxs-user";      $typeInput = "text";    $placeholderInput = "Username"; include '../Components/control.php'; ?>
        <?php $isDisabled = false; $name = "emailSignup";    $icon = "bx bxs-envelope";  $typeInput = "email";   $placeholderInput = "Email";    include '../Components/control.php'; ?>
        <?php $isDisabled = false; $name = "passwordSignup"; $icon = "bx bxs-lock";      $typeInput = "password";$placeholderInput = "Password"; include '../Components/control.php'; ?>
        <?php $nameButton = "btnSignup";$textButton = "Registrarte"; $isTransparent = false; include '../Components/button.php'; ?>
      </form>
    </div>

    <div class="container--right">
      <div class="panel block" id="panel--sign-in">
        <div class="content">
          <h3>¿ Eres uno de nosotros ?</h3>
          <p> Si ya tienes una cuenta ingresa al botón de abajo.</p>
          <?php $nameButton = "sign-in-btn"; $textButton = "Iniciar Sesión";  $isTransparent = true; include '../Components/button.php'; ?>
        </div>
        <div class="image image--right"></div>
      </div>

      <form action="../../Controllers/loginController.php" method="POST" id="form--sign-up" class="form">
        <h2 class="title">Iniciar Sesión</h2>
        <?php $isDisabled = false; $name = "usernameSignin"; $icon = "bx bxs-user";  $typeInput = "text";    $placeholderInput = "Username"; include '../Components/control.php'; ?>
        <?php $isDisabled = false; $name = "passwordSignin"; $icon = "bx bxs-lock";  $typeInput = "password";$placeholderInput = "Password"; include '../Components/control.php'; ?>
        <?php $nameButton = "btnSignin"; $textButton = "Iniciar sesión";  $isTransparent = false;   include '../Components/button.php'; ?>
        <a href="" class="form__link">Olvidé mi contraseña</a>
      </form>
    </div>
  </div>

  <div class="alert block" id="loginAlert">
    <div>
      <i class='bx bxs-x-circle' id="closeLoginAlert"></i>
      <span></span>
    </div>  
  </div>

  <script src="../Scripts/login.js" type="module"></script>
</body>
</html>