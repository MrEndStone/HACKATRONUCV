import { ajaxGet } from "./ajax.js";

document.addEventListener('DOMContentLoaded', () => {
    const detailsClose = document.getElementById('closeServicesDetails');
    detailsClose.addEventListener('click', () => {
        const detailsContainer = document.getElementById('servicesDetails');
        detailsContainer.classList.add('block')
    })

    getReservations();

    const btnNewReserve = document.getElementById('newReserva');
    btnNewReserve.addEventListener('click', () => window.location.replace("http://localhost/Projects/RapiReserva/Views/Pages/app.php?page=nuevaReserva"));
})

async function getReservations() {
    let userId = localStorage.getItem('userId') || 0;
    const url = `http://localhost/Projects/RapiReserva/Controllers/reservationsController.php?getReservations=true&userId=${userId}`;
    const response = await ajaxGet(url);
    writeTable(response);
}

function writeTable(reservations) {
    const tableBody = document.querySelector('.data tbody');
    
    reservations.forEach(reservation => {
        const row = document.createElement('tr');
        row.innerHTML = `<td>${reservation.startDate}</td>
                        <td>${reservation.status}</td>
                        <td><a class="showDetails" href="#" id="details-${reservation.id}">Ver más</a></td>`;
        row.innerHTML += '</tr>';              
        tableBody.appendChild(row);
    });
    bindEventDetails();
}

function bindEventDetails() {
    const details = document.querySelectorAll('.showDetails');
    details.forEach(detail => {
       detail.addEventListener('click', getServices); 
    });
}

function showModal(data) {
    const detailsContainer = document.getElementById('servicesDetails');
    const tableBodyDetails = document.querySelector('#servicesDetails tbody');
    data.forEach(service => {
        const row = document.createElement('tr');
        row.innerHTML = `<td>${service.description}</td>
                        <td>${(service.category).toLowerCase()}</td>
                        <td>${service.duration} min.</td>
                        <td>${service.price} soles</td>`;
        row.innerHTML += '</tr>';              
        tableBodyDetails.appendChild(row);
    });
    detailsContainer.classList.remove('block');
}

async function getServices(ev) {
    ev.preventDefault();
    let id = ev.target.id.split('-')[1]; 
    const url = `http://localhost/Projects/RapiReserva/Controllers/reservationsController.php?getServices=true&reservationId=${id}`;
    const response = await ajaxGet(url);
    showModal(response);
}
