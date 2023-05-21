<link rel="stylesheet" href="../Css/perfil.css">

<section class="perfil">
    <form action="">
        <fieldset>
            <legend>Datos Generales</legend>
            <?php $isDisabled = false; $name = "fullnamePerfil"; $icon = "bx bxs-user";  $typeInput = "text"; $placeholderInput = "Nombre completo";  include '../Components/control.php'; ?>
            <?php $isDisabled = true; $name = "emailPerfil";    $icon = "bx bxs-user";  $typeInput = "email"; $placeholderInput = "Correo";           include '../Components/control.php'; ?>
            <?php $isDisabled = false; $name = "dniPerfil";      $icon = "bx bxs-user";  $typeInput = "text"; $placeholderInput = "Doc. de identidad";include '../Components/control.php'; ?>
        </fieldset>

        <fieldset>
            <legend>Datos Personales</legend>
            <?php $isDisabled = false; $name = "phonePerfil"; $icon = "bx bxs-user";  $typeInput = "text"; $placeholderInput = "Teléfono"; include '../Components/control.php'; ?>
            <?php $isDisabled = false; $name = "cityPerfil";    $icon = "bx bxs-user";  $typeInput = "text"; $placeholderInput = "Distrito"; include '../Components/control.php'; ?>
            <?php $isDisabled = false; $name = "addressPerfil";      $icon = "bx bxs-user";  $typeInput = "text"; $placeholderInput = "Dirección";include '../Components/control.php'; ?>
        </fieldset>

        <div class="row--right">
            <?php $nameButton = "updatePerfil"; $textButton = "Guardar cambios"; $isTransparent = false; include '../Components/button.php'; ?>
        </div>
    </form>
</section>

<script src="../Scripts/perfil.js" type="module"></script>