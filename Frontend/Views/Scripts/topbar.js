document.addEventListener('DOMContentLoaded', (ev) => {
    getDataUser();
})

function getDataUser() {
    const user = document.querySelector('.header__username');
    const photo = document.querySelector('.header__photo > img');

    let userName = localStorage.getItem('userName') || 'Username';
    let userUrlPhoto = (localStorage.getItem('urlPhoto') != 'null') ?  localStorage.getItem('urlPhoto') : '../Assets/user-default.png';

    user.textContent = userName;
    photo.src = userUrlPhoto;
}