document.addEventListener('DOMContentLoaded', () => {

})

const btnConfirm = document.getElementById('btnConfirmPayment');
btnConfirm.addEventListener('click', confirmPayment);
const inputCardNumber = document.getElementById('cardNumber');
inputCardNumber.addEventListener('keyup', writeCardNumber);
const inputCodeSec = document.getElementById('cardCodSec');
inputCodeSec.addEventListener('keyup', verifyPaymentDetails);
const inputDateExp = document.getElementById('cardDateExp');
inputDateExp.addEventListener('keyup', writeDate);
inputDateExp.addEventListener('change', writeDate);
const inputCardOwner = document.getElementById('cardOwner');
inputCardOwner.addEventListener('keyup', verifyPaymentDetails);
const imageCard = document.querySelector('.imageCard');

function writeCardNumber() {
    let paymentDetails = verifyPaymentDetails();
    if(paymentDetails.cardNumber != null) {
        let card = paymentDetails.cardNumber;
        for (let i = 0; i < 4; i++) {
            let number = card.slice(i*4, (i+1)*4);
            imageCard.querySelector('.imageCard__numbers--'+i).textContent = number;
        }
    }
}

function writeDate() {
    let paymentDetails = verifyPaymentDetails();
    if(paymentDetails.dateExp != null) {
        const dateContent = imageCard.querySelector('.imageCard__date');
        let date = paymentDetails.dateExp.split('-');
        let year = date[0];
        let month = date[1];
        dateContent.textContent = `${month}/${year.slice(2,4)}`;
    }
}

function confirmPayment() {
    let paymentDetails = verifyPaymentDetails();
    if(paymentDetails.isValid) {
        localStorage.setItem('dataPayment', JSON.stringify(paymentDetails));
        window.location.replace("http://localhost/Projects/RapiReserva/Views/Pages/app.php?page=verificar");
    } else {
        console.log('Completa los datos de formulario');
    }
}

function verifyPaymentDetails() {
    let paymentDetails = {isValid : true};
    const typeCard = document.querySelector('input[name="typeCard"]:checked');
    paymentDetails.typeCard = typeCard != null ? typeCard.value : null;
    const cardNumber = document.getElementById('cardNumber').value;
    paymentDetails.cardNumber = cardNumber.length === 16 ? cardNumber : null;
    const codeSec = document.getElementById('cardCodSec').value;
    paymentDetails.codeSec = codeSec.length === 3 ? codeSec : null;
    const cardDateExp = document.getElementById('cardDateExp').value;
    paymentDetails.dateExp = cardDateExp != '' ? cardDateExp : null;
    const cardOwner = document.getElementById('cardOwner').value;
    let regex = /^[A-Za-z\s]*$/;
    paymentDetails.cardOwner = cardOwner != '' && cardOwner.match(regex) ? cardOwner : null;
    for (const key in paymentDetails) {
        if(paymentDetails[key] == null) {
            paymentDetails.isValid = false;
        }
    }

    if(paymentDetails.isValid) {
        document.getElementById('btnConfirmPayment').classList.remove('disabled');
    }
    return paymentDetails;
}
