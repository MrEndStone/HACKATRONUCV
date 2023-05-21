import { ajaxGet, ajaxPost } from "./ajax.js";

document.addEventListener('DOMContentLoaded', () => {
    const alertClose = document.getElementById('closeLoginAlert');
    alertClose.addEventListener('click', () => {
        const alertContainer = document.getElementById('appAlert');
        alertContainer.classList.add('block')
    })

    getUserProfile();

    const btnUpdateProfile = document.getElementById('updatePerfil');
    btnUpdateProfile.addEventListener('click', updateProfile);
})

function updateProfile(ev) {
    ev.preventDefault();
    const fullname = document.getElementById('fullnamePerfil').value;
    const dni = document.getElementById('dniPerfil').value;
    const phone = document.getElementById('phonePerfil').value;
    const city = document.getElementById('cityPerfil').value;
    const address = document.getElementById('addressPerfil').value;

    let user = {
        userId : localStorage.getItem('userId') || 0,
        fullnamePerfil : fullname,
        dniPerfil : dni,
        phonePerfil : phone,
        addressPerfil : address,
        cityPerfil : city
    }
    updateUser(user);
}

async function getUserProfile() {
    let userId = localStorage.getItem('userId') || 0;
    const url = `http://localhost/Projects/RapiReserva/Controllers/perfilController.php?getPerfil=true&userId=${userId}`;
    const response = await ajaxGet(url);
    writeInformation(response);
}

async function updateUser(userData) {
    const url = 'http://localhost/Projects/RapiReserva/Controllers/perfilController.php';
    let values = {
        ...userData,
        updatePerfil: true
    }
    const form = new FormData();
    for(const property in values) {
        form.append(property, values[property])
    }
    const response = await ajaxPost(url, values);
    const alertContainer = document.getElementById('appAlert');
    if(response) {
        alertContainer.firstElementChild.lastElementChild.textContent = 'Perfil actualizado con éxito.';
        alertContainer.firstElementChild.classList.add('alert--success');    
    } else {
        alertContainer.firstElementChild.lastElementChild.textContent = 'Ocurrió un error al actualizar los datos.';
        alertContainer.firstElementChild.classList.add('alert--error');    
    }
    alertContainer.classList.remove('block');
}


function writeInformation(user) {
    const fullname = document.getElementById('fullnamePerfil');
    const email = document.getElementById('emailPerfil');
    const dni = document.getElementById('dniPerfil');
    const phone = document.getElementById('phonePerfil');
    const city = document.getElementById('cityPerfil');
    const address = document.getElementById('addressPerfil');
    
    fullname.value = user.fullname ?? '';
    email.value = user.email ?? '';
    dni.value = user.dni ?? '';
    phone.value = user.phone ?? '';
    city.value = user.city ?? '';
    address.value = user.address ?? '';
}
