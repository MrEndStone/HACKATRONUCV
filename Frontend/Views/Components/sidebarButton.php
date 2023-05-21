<link rel="stylesheet" href="../Css/sidebarButton.css">

<a 
    href = "<?= isset($url) ? $url : "" ?>"
    class="btn--sidebar <?= isset($active) ? $active : "" ?>" >
        <i class='<?= isset($icon) ? $icon : ""  ?>'></i>
        <span><?= isset($text) ? $text : "" ?></span>
</a>