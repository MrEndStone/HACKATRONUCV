import { ajaxPost } from './ajax.js';

export const validateToken = async () => {
    const url = 'http://localhost/Projects/RapiReserva/Controllers/authController.php';
    let token = localStorage.getItem('tokenAuth') || "";
    let userId = localStorage.getItem('userId') || 0;
    let values = {
        token: token,
        userId: userId,
        validate: true
    }
    const form = new FormData();
    for(const property in values) {
        form.append(property, values[property])
    }
    const response = await ajaxPost(url, values);
    localStorage.setItem('tokenAuth', response.token);
    localStorage.setItem('userId', response.userId);
    if(response.isValid && !window.location.href.includes('http://localhost/Projects/RapiReserva/Views/Pages/app.php')) {
        window.location.replace("http://localhost/Projects/RapiReserva/Views/Pages/app.php");
    } 
    if(!response.isValid && !window.location.href.includes('http://localhost/Projects/RapiReserva/Views/Pages/login.php')) {
        localStorage.clear();
        window.location.replace("http://localhost/Projects/RapiReserva/Views/Pages/login.php");
    }
}

export async function saveToken(userId) {
    const url = 'http://localhost/Projects/RapiReserva/Controllers/authController.php';
    const values = {
      userId : userId,
      saveAuth: true 
    };
    const data = await ajaxPost(url, values);
    if(data.isValid) {
        localStorage.setItem('tokenAuth', data.token);
        localStorage.setItem('userId', data.userId);
        return true;
    }
    return false;
  }