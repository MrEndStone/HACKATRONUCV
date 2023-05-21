let calendar = document.querySelector('.calendar')

const month_names = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']

function isLeapYear(year) {
    return (year % 4 === 0 && year % 100 !== 0 && year % 400 !== 0) || (year % 100 === 0 && year % 400 ===0)
}

function getFebDays(year) {
    return isLeapYear(year) ? 29 : 28
}

function generateCalendar(month, year) {

    let calendar_days = calendar.querySelector('.calendar-days')
    let calendar_header_year = calendar.querySelector('#year')

    let days_of_month = [31, getFebDays(year), 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    calendar_days.innerHTML = ''

    let currDate = new Date()
    if (!month) month = currDate.getMonth()
    if (!year) year = currDate.getFullYear()

    let curr_month = `${month_names[month]}`
    month_picker.innerHTML = curr_month
    calendar_header_year.innerHTML = year

    // get first day of month
    
    let first_day = new Date(year, month, 1)

    for (let i = 0; i <= days_of_month[month] + first_day.getDay() - 1; i++) {
        let day = document.createElement('div')
        if (i >= first_day.getDay()) {
            day.classList.add('calendar-day-hover')
            day.innerHTML = i - first_day.getDay() + 1
            day.innerHTML += `<span></span>
                            <span></span>
                            <span></span>
                            <span></span>`
            if (i - first_day.getDay() + 1 === currDate.getDate() && year === currDate.getFullYear() && month === currDate.getMonth()) {
                day.classList.add('curr-date')
            }
        }
        calendar_days.appendChild(day)
        bindEventDay();
    }
}

let month_list = calendar.querySelector('.month-list')

month_names.forEach((e, index) => {
    let month = document.createElement('div')
    month.innerHTML = `<div data-month="${index}">${e}</div>`
    month.querySelector('div').onclick = () => {
        month_list.classList.remove('show')
        curr_month.value = index
        generateCalendar(index, curr_year.value)
    }
    month_list.appendChild(month)
    
})

let month_picker = calendar.querySelector('#month-picker')

month_picker.onclick = () => {
    month_list.classList.add('show')
}

let currDate = new Date()

let curr_month = {value: currDate.getMonth()}
let curr_year = {value: currDate.getFullYear()}

generateCalendar(curr_month.value, curr_year.value)

document.querySelector('#prev-year').onclick = () => {
    --curr_year.value
    generateCalendar(curr_month.value, curr_year.value)
}

document.querySelector('#next-year').onclick = () => {
    ++curr_year.value
    generateCalendar(curr_month.value, curr_year.value)
}

document.addEventListener('DOMContentLoaded', () => {
    loadDate();
    bindEventDay();

    const btnConfirm = document.getElementById('btnConfirmReserve');
    btnConfirm.addEventListener('click', confirmReserve);
})

function loadDate() {
    const year = document.getElementById('year').innerText;
    const month = document.getElementById('month-picker').innerText;
    const day = document.querySelector('.curr-date').innerText;
    
    let date = new Date(year, month_names.indexOf(month), day);
    localStorage.setItem('dateForReserve', date);
}

function bindEventDay() {
    const days = document.querySelectorAll('.calendar-day-hover');
    days.forEach(day => day.addEventListener('click', ev => {
        days.forEach(child => child.classList.remove('curr-date'));
        ev.target.classList.add('curr-date');
        loadDate();
    }))
}

function confirmReserve() {
    let cart = localStorage.getItem('cart');
    if(cart == null || cart == '[}' || cart == '') {
        console.log('Debe agregar items al carrito');
        return;
    } 
    let date = localStorage.getItem('dateForReserve');
    if(date == null || date == '') {
        console.log('Debe seleccionar una fecha para la reserva');
        return;
    }
    window.location.replace("http://localhost/Projects/RapiReserva/Views/Pages/app.php?page=pago");
}
