import { ajaxGet } from "./ajax.js";

document.addEventListener('DOMContentLoaded', () => {
    getPayments();
})

async function getPayments() {
    let userId = localStorage.getItem('userId') || 0;
    const url = `http://localhost/Projects/RapiReserva/Controllers/paymentsController.php?getPayments=true&userId=${userId}`;
    const response = await ajaxGet(url);
    writeTable(response);
}

function writeTable(payments) {
    const tableBody = document.querySelector('.data tbody');
    
    payments.forEach(payment => {
        const row = document.createElement('tr');
        row.innerHTML = `<td>${payment.voucherCode}</td>
                        <td>${payment.paymentDate}</td>
                        <td>${payment.reservaPrice} soles</td>
                        <td><a href="http://localhost/Projects/RapiReserva/Controllers/paymentsController.php?getPaymentPdf=true&paymentId=${payment.id}" target="_blank"><i class='bx bxs-download'></i></a></td>`;
        row.innerHTML += '</tr>'; 
        tableBody.appendChild(row);
    });
    console.log(payments);
}
