import { ajaxPost } from './ajax.js';
import { saveToken, validateToken } from './auth.js';

const formSignin = document.querySelector("#form--sign-in");
const formSignup = document.querySelector("#form--sign-up");

const sign_in_btn = document.querySelector("#sign-in-btn");
const sign_up_btn = document.querySelector("#sign-up-btn");

const container = document.querySelector(".container");
const alertContainer = document.getElementById('loginAlert');
const alertClose = document.getElementById('closeLoginAlert');

sign_up_btn.addEventListener("click", () => {
  container.classList.remove("sign-in");
  container.classList.add("sign-up");
  document.querySelector("#panel--sign-in").classList.remove("block");
  formSignin.classList.remove("block");
  document.querySelector("#panel--sign-up").classList.add("block");
  formSignup.classList.add("block");
});

sign_in_btn.addEventListener("click", () => {
  container.classList.remove("sign-up");
  container.classList.add("sign-in");
  document.querySelector("#panel--sign-in").classList.add("block");
  document.querySelector("#form--sign-in").classList.add("block");
  document.querySelector("#panel--sign-up").classList.remove("block");
  document.querySelector("#form--sign-up").classList.remove("block");
});

alertClose.addEventListener('click', (ev) => alertContainer.classList.add('block'))

formSignin.addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const url = 'http://localhost/Projects/RapiReserva/Controllers/loginController.php';
  const values = {
    usernameSignup : document.getElementById('usernameSignup').value,
    emailSignup : document.getElementById('emailSignup').value,
    passwordSignup : document.getElementById('passwordSignup').value,
    btnSignup: true 
  };
  const data = await ajaxPost(url, values);
  if(data.success) {
    formSignin.reset();
    alertContainer.firstElementChild.lastElementChild.textContent = 'Cuenta registrada con éxito.';
    alertContainer.firstElementChild.classList.add('alert--success');
  } else {
    alertContainer.firstElementChild.lastElementChild.textContent = 'Ocurrió un error en el registro.';
    alertContainer.firstElementChild.classList.add('alert--error');
  }
  alertContainer.classList.remove('block');
})

formSignup.addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const url = 'http://localhost/Projects/RapiReserva/Controllers/loginController.php';
  const values = {
    usernameSignin : document.getElementById('usernameSignin').value,
    passwordSignin : document.getElementById('passwordSignin').value,
    btnSignin: true 
  };
  const data = await ajaxPost(url, values);
  if(data.Valid) {
    localStorage.setItem('userName', data.Username);
    localStorage.setItem('urlPhoto', data.UrlAvatar);
    formSignup.reset();
    const response = await saveToken(data.Id);
    if(response) {
      window.location.replace("http://localhost/Projects/RapiReserva/Views/Pages/app.php");
    } else {
      alertContainer.firstElementChild.lastElementChild.textContent = 'No se pudo guardar el token de ingreso.';
      alertContainer.firstElementChild.classList.add('alert--error');
      alertContainer.classList.remove('block');
    }
  } else {
    alertContainer.firstElementChild.lastElementChild.textContent = 'Los datos de acceso no son válidos.';
    alertContainer.firstElementChild.classList.add('alert--error');
    alertContainer.classList.remove('block');
  }
})

document.addEventListener('DOMContentLoaded', (ev) => {
  validateToken();
})