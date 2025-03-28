window.addEventListener("DOMContentLoaded", function () {
  var modalOverlay = document.getElementById("modal-overlay");
  var acceptBtn = document.getElementById("accept-btn");
  acceptBtn.addEventListener("click", function () {
    modalOverlay.style.display = "none";
    document.body.style.overflow = "auto";
  });
  document.body.style.overflow = "hidden";
  var form = document.getElementById("riskForm");
  form.addEventListener("submit", function (event) {
    event.preventDefault();
    var formData = new FormData(form);
    fetch("/api/risk", { method: "POST", body: formData })
      .then((response) => response.json())
      .then((data) => {
        var resultDiv = document.getElementById("risk-result");
        if (data.errors) {
          resultDiv.innerHTML = "";
          Object.keys(data.errors).forEach(function (key) {
            resultDiv.innerHTML +=
              "<p>" + key + ": " + data.errors[key] + "</p>";
          });
        } else {
          resultDiv.innerHTML =
            "<h2 style='color:" +
            data.color +
            "'>Ваш ризик: " +
            data.level +
            "</h2>";
          document.getElementById("risk_result_input").value = data.level;
        }
      });
  });
});
function resetForm() {
  document.getElementById("riskForm").reset();
  document.getElementById("risk-result").innerHTML = "";
  document.getElementById("risk_result_input").value = "";
}
