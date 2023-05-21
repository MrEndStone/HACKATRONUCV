import { ajaxGet } from './ajax.js';
import { getPromotions } from './slider.js';

document.addEventListener('DOMContentLoaded', (ev) => {
    const contenedor = document.getElementById('sliderContainer');
    getPromotions(contenedor);
    getLastReservation();
})

function writeLastReservation(date) {
    const content = document.getElementById('reservationDays');
    if(date) {
        let reservationDate = new Date(date);
        let currentDate = new Date();
        let difference = currentDate.getTime() - reservationDate.getTime();
        let days = Math.ceil(difference / (1000 * 3600 * 24));
        if(days == 0) {
            content.textContent = `Tu última reserva fue hoy`;
        }
        if(days == 1) {
            content.textContent = `Tu última reserva fue hace ${days} día`;
        }
        if(days > 1) {
            content.textContent = `Tu última reserva fue hace ${days} días`;
        }
    } else {
        content.textContent = `Aún no has realizado tu primera reserva`;
    }
}

async function getLastReservation() {
    let userId = localStorage.getItem('userId') || 0;
    const url = `http://localhost/Projects/RapiReserva/Controllers/reservationsController.php?getLastReservation=true&userId=${userId}`;
    const response = await ajaxGet(url);
    writeLastReservation(response.startDate);
}