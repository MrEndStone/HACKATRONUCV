<link rel="stylesheet" href="../Css/calendar.css">
<link rel="stylesheet" href="../Css/button.css">

<section class="sectionCalendar">
    <div class="sectionCalendar--header">
        <a class="btn" href="http://localhost/Projects/RapiReserva/Views/Pages/app.php?page=nuevaReserva">Volver</a>
        <input class="btn" type="button" value="Continuar" id="btnConfirmReserve">
    </div>

    <div class="calendar">
        <div class="calendar-header">
            <span class="month-picker btn" id="month-picker">February</span>
            <div class="year-picker btn">
                <span class="year-change" id="prev-year">
                    <pre><</pre>
                </span>
                <span id="year">2021</span>
                <span class="year-change" id="next-year">
                    <pre>></pre>
                </span>
            </div>
        </div>
        <div class="calendar-body">
            <div class="calendar-week-day">
                <div>Sun</div>
                <div>Mon</div>
                <div>Tue</div>
                <div>Wed</div>
                <div>Thu</div>
                <div>Fri</div>
                <div>Sat</div>
            </div>
            <div class="calendar-days"></div>
        </div>
        <div class="month-list"></div>
    </div>
</section>

<script src="../Scripts/calendar.js" type="module"></script>