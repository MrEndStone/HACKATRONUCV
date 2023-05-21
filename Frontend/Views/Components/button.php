<link rel="stylesheet" href="../Css/button.css">

<button 
    class="btn <?= $isTransparent ? "transparent" : ""  ?>" 
    type="submit"
    id="<?= $nameButton ?>"
    name="<?= $nameButton ?>">
        <?= $textButton ?>
</button>