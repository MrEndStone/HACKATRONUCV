<link rel="stylesheet" href="../Css/icon.css">

<div id="<?= $id ?>" class="icon <?= isset($className) ? $className : '' ?> ">
    <?php if(isset($icon)) { ?>
        <i class='<?= $icon ?>'></i>
    <?php } ?>
    <?php if(isset($imageUrl)) { ?>
        <img src="<?= $imageUrl ?>" alt="">
    <?php } ?>
</div>