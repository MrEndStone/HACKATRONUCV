import { ajaxPost } from './ajax.js';

document.addEventListener('DOMContentLoaded', () => {
    loadData();
})

const btnConfirm = document.getElementById('confirmReserve');
btnConfirm.addEventListener('click', submitReserve)

function loadServices() {
    const servicesContainer = document.querySelector('.verification__services--body');
    const services = JSON.parse(localStorage.getItem('cart')) ?? [];
    let displayItems = services.map(service => {
        return `
            <div class="verification__services--item">
                <div class="verification__services--image">
                    <img src="${service.urlImage}" alt="">
                </div>
                <p class="verification__services--bold">${service.description}</p>
                <p>${service.price} soles</p>
                <p>${service.duration} min</p>
            </div>
        `
    })
    displayItems = displayItems.join("");
    servicesContainer.innerHTML = displayItems;
    loadResumen(services);
}

function loadResumen(services) {
    let subtotal = 0;
    let duration = 0;
    services.forEach(item => {
        subtotal += parseInt(item.price);
        duration += parseInt(item.duration);
    })
    const subTotalPriceText = document.getElementById('verificationPriceSubtotal');
    const reservaPriceText = document.getElementById('verificationPriceReserva');
    const totalPriceText = document.getElementById('verificationPriceTotal');
    const totalDurationText = document.getElementById('verificationDurationTotal');
    subTotalPriceText.textContent = (subtotal).toFixed(2) + " soles";
    reservaPriceText.textContent = (subtotal*0.1).toFixed(2) + " soles";
    totalPriceText.textContent = (subtotal*1.1).toFixed(2) + " soles";
    totalDurationText.textContent = duration + " min";
}

function loadPaymentMethod() {
    let paymentDetails = JSON.parse(localStorage.getItem('dataPayment'));
    if(paymentDetails.typeCard != null) {
        const cardImage = document.getElementById(paymentDetails.typeCard+'Card');
        cardImage.classList.remove('hidden');
    }
}

function loadDateOfReserve() {
    let date = new Date(localStorage.getItem('dateForReserve'));
    if(date != '') {
        const dateText = document.querySelector('.verification__date');
        dateText.textContent = date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
    }
}

function loadData() {
    loadServices();
    loadPaymentMethod();
    loadDateOfReserve();
}

async function submitReserve(ev) {
    ev.preventDefault();
    let services = localStorage.getItem('cart');
    let paymentDetails = localStorage.getItem('dataPayment');
    let dateString = new Date(localStorage.getItem('dateForReserve'));
    let dateReserve = dateString.toLocaleDateString();
    let hourReserve = dateString.toLocaleTimeString();
    let userId = localStorage.getItem('userId');
    let token = localStorage.getItem('tokenAuth');
    if(services != '' && paymentDetails != '' && userId != '' && token != '') {
        const url = 'http://localhost/Projects/RapiReserva/Controllers/reservationsController.php';
        const values = {
            services : services,
            payment : paymentDetails,
            date : dateReserve,
            hour : hourReserve,
            userId : userId,
            token : token,
            registerReservation: true 
        };
        const data = await ajaxPost(url, values);
        if(data.success) {
            localStorage.setItem('dateForReserve', '');
            localStorage.setItem('dataPayment', '');
            localStorage.setItem('cart', '[]');
            window.location.replace("http://localhost/Projects/RapiReserva/Views/Pages/app.php?page=reservas");
        } else {
            console.log('No se pudo registrar la reserva.');
        }
    }
}