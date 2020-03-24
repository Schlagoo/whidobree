function close_overlay() {
    /**
     * Function to handle cookies overlay
     */

    var overlay = document.getElementById("overlay");
    var message = document.getElementById("cookie-message");

    if (message.style.display !== "none") {
        message.style.display = "none";
        overlay.style.display = "none";
    }
}