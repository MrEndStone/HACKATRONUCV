<link rel="stylesheet" href="../Css/control.css">

<div class="control">
    <i class="<?= $icon ?>"></i>
    <input type="<?= $typeInput ?>" placeholder="<?= $placeholderInput ?>" id="<?= $name ?>" name="<?= $name ?>" <?= $isDisabled ? 'disabled' : '' ?> />
</div>