import { ajaxGet } from "./ajax.js";

class IndexForSiblings {
  static get(el) {
    let children = el.parentNode.children;
    for (let i = 0; i < children.length; i++) {
      let child = children[i];
      if (child == el) {
        return i;
      }
    }
  }
}

class Slider {
  constructor(selector) {
    this.move = this.move.bind(this);
    this.moveByButton = this.moveByButton.bind(this);
    this.slider = document.querySelector(selector);
    this.itemsCount = this.slider.querySelectorAll(
      ".slider-container > *"
    ).length;
    this.interval = null;
    this.counter = 0;

    this.start();
    this.buildControls();
    this.bindEvents();
  }

  start() {
    this.interval = window.setInterval(this.move, 3000);
  }

  restart() {
    if (this.interval) window.clearInterval(this.interval);
    this.start();
  }

  bindEvents() {
    this.slider.querySelectorAll(".controls li").forEach((item) => {
      item.addEventListener("click", this.moveByButton);
    });
  }

  moveByButton(ev) {
    let index = IndexForSiblings.get(ev.currentTarget);
    this.counter = index;
    this.moveTo(index);
    this.restart();
  }

  buildControls() {
    for (let i = 0; i < this.itemsCount; i++) {
      let control = document.createElement("li");
      if (i == 0) {
        control.classList.add("active");
      }
      this.slider.querySelector(".controls ul").appendChild(control);
    }
  }

  move() {
    this.counter++;
    if (this.counter >= this.itemsCount) {
      this.counter = 0;
    }
    this.moveTo(this.counter);
  }

  resetIndicator() {
    this.slider
      .querySelectorAll(".controls li.active")
      .forEach((item) => item.classList.remove("active"));
  }

  moveTo(index) {
    let left = index * 100;
    this.resetIndicator();
    this.slider
      .querySelector(".controls li:nth-child(" + (index + 1) + ")")
      .classList.add("active");
    this.slider.querySelector(".slider-container").style.left =
      "-" + left + "%";
  }
}

export async function getPromotions(contenedor) {
  const url = 'http://localhost/Projects/RapiReserva/Controllers/salesController.php?getSales=All';
  const response = await ajaxGet(url);
  response.forEach((sale) => {
    let img = document.createElement('img');
    img.src = sale.urlBanner;
    contenedor.appendChild(img);
  })
  new Slider(".slider");
}
