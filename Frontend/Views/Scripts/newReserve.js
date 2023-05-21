import { ajaxGet } from './ajax.js';

document.addEventListener('DOMContentLoaded', () => {    
    getServices();
    getCategories();
    refreshCart();
    loadUserCart();

    const cartIcon = document.getElementById('cartIcon');
    cartIcon.addEventListener('click', displayCart);

    const closeCart = document.getElementById('closeCart');
    closeCart.addEventListener('click', hideCart);
})

async function getCategories() {
    const url = `http://localhost/Projects/RapiReserva/Controllers/categoriesController.php?newReserve=true&getCategories=true`;
    const response = await ajaxGet(url);
    displayFiltersButtons(response);
}

function displayFiltersButtons(categories) {
    const filtersContainer = document.getElementById('catalogoFilters');
    categories = [...categories, {id: 0, description: 'TODO'}];
    let categoryBtns = categories.map(category => `<button class="filter__btn" id="category-${category.id}">${category.description}</button>`).join("")
    filtersContainer.innerHTML = categoryBtns
    
    const btnsFilter = filtersContainer.querySelectorAll('.filter__btn');
    btnsFilter.forEach(btn => {
        btn.addEventListener('click', ev => {
            ev.target.parentElement.childNodes.forEach(child => child.classList.remove('selected'));
            ev.target.classList.add('selected');
            let categoryId = ev.target.id.split('-')[1];
            let services = JSON.parse(localStorage.getItem('servicesList'));
            if(categoryId == 0) {
                displayItems(services)
            } else {
                const servicesFiltered = services.filter(service => service.category === categoryId)
                displayItems(servicesFiltered)
            }
        })
    })
}

async function getServices() {
    const url = `http://localhost/Projects/RapiReserva/Controllers/reservationsController.php?newReserve=true&getServices=true`;
    const response = await ajaxGet(url);
    localStorage.setItem('servicesList', JSON.stringify(response));
    displayItems(response);
}

function displayItems(services) {
    const servicesContainer = document.getElementById('catalogoServices');
    let displayItems = services.map(service => {
        return `
            <article class="service" id="service-${service.id}">
                <div class="service__image">
                    <img src="${service.urlImage}" alt="">
                </div>
                <div class="service__info">
                    <h3 class="service__name">${service.description}</h3>
                    <p>Precio base: ${service.price}</p>
                    <p>Duración mínima: ${service.duration} min</p>
                    <button class="btn agregar-service">Agregar</button>
                </div>
            </article>
        `
    })

    displayItems = displayItems.join("")
    servicesContainer.innerHTML = displayItems

    const btnsAddService = servicesContainer.querySelectorAll('.agregar-service');
    btnsAddService.forEach(btn => btn.addEventListener('click', addService));
}

function displayCart() {
    const bgCart = document.querySelector('.cart--bg');
    bgCart.classList.add('open');
}

function hideCart() {
    const bgCart = document.querySelector('.cart--bg');
    bgCart.classList.remove('open');
}

function addService(e) {
    let serviceId = e.target.parentElement.parentElement.id.split('-')[1];
    const services = JSON.parse(localStorage.getItem('servicesList'));
    let service = services.filter(item => item.id == serviceId)[0];
    const cart = JSON.parse(localStorage.getItem('cart')) ?? [];
    cart.push(service);
    localStorage.setItem('cart', JSON.stringify(cart));
    refreshCart();
}

function refreshCart() {
    const cartContainer = document.querySelector('.cart__details--body');
    const cart = JSON.parse(localStorage.getItem('cart')) ?? [];
    let displayItems = cart.map(service => {
        return `
            <div class="cart__details--item">
                <div class="cart__item--image">
                    <img src="${service.urlImage}" alt="">
                </div>
                <h3 class="cart__item--bold">${service.description}</h3>
                <p>${service.price}</p>
                <p>${service.duration} min</p>
                <div><i class='bx bxs-trash eliminar-service' id="itemCart-${service.id}"></i></div>
            </div>
        `
    })
    displayItems = displayItems.join("")
    cartContainer.innerHTML = displayItems

    const btnsAddService = cartContainer.querySelectorAll('.eliminar-service');
    btnsAddService.forEach(btn => btn.addEventListener('click', deleteService));

    loadTotalValues(cart);
}

function loadUserCart() {
    const userText = document.querySelector('.cart__username');
    let username = localStorage.getItem('userName') || '';
    userText.textContent = username;
}

function loadTotalValues(cart) {
    const totalPriceText = document.querySelector('.cart__price--foot');
    const totalDurationText = document.querySelector('.cart__duration--foot');
    let price = 0;
    let duration = 0;
    cart.forEach(item => {
        price += parseInt(item.price);
        duration += parseInt(item.duration);
    })
    totalPriceText.textContent = price;
    totalDurationText.textContent = duration + " min";
}

function deleteService(e) {
    let serviceId = e.target.id.split('-')[1];
    let cart = JSON.parse(localStorage.getItem('cart'));
    cart = cart.filter(item => item.id != serviceId);
    localStorage.setItem('cart', JSON.stringify(cart));
    refreshCart();
}