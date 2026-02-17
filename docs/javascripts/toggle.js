document.addEventListener("DOMContentLoaded", function () {
    // Find all toggle containers
    const toggles = document.querySelectorAll(".api-toggle");

    toggles.forEach(toggle => {
        const buttons = toggle.querySelectorAll(".toggle-btn");
        const contents = toggle.querySelectorAll(".toggle-content");
        const slider = toggle.querySelector(".toggle-slider");

        buttons.forEach((btn, index) => {
            btn.addEventListener("click", () => {
                // Update active state
                buttons.forEach(b => b.classList.remove("active"));
                btn.classList.add("active");

                // Move slider
                if (slider) {
                    slider.style.transform = `translateX(${index * 100}%)`;
                }

                // Show content
                contents.forEach(c => c.classList.remove("active"));
                contents[index].classList.add("active");
            });
        });
    });
});
